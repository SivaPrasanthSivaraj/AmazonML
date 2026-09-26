"""One-command, resumable train-to-submission production pipeline."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from .pipeline_state import PipelineState, atomic_write_json
from .production_ann import generate_ann_candidates
from .production_candidates import generate_classical_candidates
from .production_data import prepare_production_data, validate_dataset
from .production_features import build_feature_shards
from .production_model import train_production_model
from .production_pair_union import build_pair_shards
from .production_submit import score_and_write_submission


def _discover_dataset() -> Path | None:
    candidates: list[Path] = []
    for root in (Path("/kaggle/input"), Path("student_resource"), Path("dataset")):
        if not root.exists():
            continue
        candidates.extend(
            path.parent.parent for path in root.rglob("train/train_source1.tsv")
        )
    unique = sorted(set(path.resolve() for path in candidates))
    return unique[0] if len(unique) == 1 else None


def _preflight(dataset_dir: Path, work_dir: Path) -> dict[str, object]:
    validate_dataset(dataset_dir)
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for the production pipeline") from error
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Select a T4 GPU (or use a CUDA AWS instance) "
            "before starting the full pipeline."
        )
    work_dir.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(work_dir)
    free_gib = disk.free / 1024**3
    if free_gib < 35:
        raise RuntimeError(
            f"only {free_gib:.1f} GiB free below {work_dir}; the full three-view "
            "pipeline requires at least 35 GiB working space. Use AWS/EBS or a "
            "larger Kaggle disk."
        )
    return {
        "dataset_dir": str(dataset_dir.resolve()),
        "work_dir": str(work_dir.resolve()),
        "gpu": torch.cuda.get_device_name(0),
        "cuda_devices": torch.cuda.device_count(),
        "free_disk_gib": round(free_gib, 2),
    }


def _clear_embedding_files(cache_dir: Path) -> None:
    if not cache_dir.exists():
        return
    for path in cache_dir.iterdir():
        if path.is_file() and path.suffix in {".npy", ".json"}:
            path.unlink()


def run_pipeline(
    *,
    dataset_dir: Path,
    work_dir: Path,
    shard_count: int = 100,
    classical_top_k: int = 30,
    ann_top_k: int = 20,
    train_negative_cap: int = 5,
    test_candidate_cap: int = 40,
    encoder_batch_size: int = 512,
    search_batch_size: int = 4096,
    nlist: int = 4096,
    nprobe: int = 64,
    faiss_device: str = "auto",
    keep_embeddings: bool = False,
) -> dict[str, object]:
    started = time.monotonic()
    preflight = _preflight(dataset_dir, work_dir)
    print(json.dumps({"preflight": preflight}, indent=2), flush=True)
    state = PipelineState(work_dir)
    normalized_dir = work_dir / "normalized"
    candidates_dir = work_dir / "candidates"
    pairs_dir = work_dir / "pair_data"
    features_dir = work_dir / "feature_data"
    model_dir = work_dir / "model_data"
    output_dir = work_dir / "output"
    cache_dir = work_dir / "embedding_cache"

    prepare_production_data(dataset_dir, normalized_dir, state)

    # Train: retrieve -> union/label -> direct features -> calibrate/refit.
    generate_classical_candidates(
        normalized_dir=normalized_dir,
        output_dir=candidates_dir,
        split="train",
        shard_count=shard_count,
        top_k=classical_top_k,
        state=state,
    )
    generate_ann_candidates(
        normalized_dir=normalized_dir,
        output_dir=candidates_dir,
        cache_dir=cache_dir,
        split="train",
        state=state,
        shard_count=shard_count,
        top_k=ann_top_k,
        encoder_batch_size=encoder_batch_size,
        search_batch_size=search_batch_size,
        index_type="ivf",
        nlist=nlist,
        nprobe=nprobe,
        encoder_device="cuda",
        faiss_device=faiss_device,
        keep_embeddings=True,
    )
    build_pair_shards(
        normalized_dir=normalized_dir,
        candidates_dir=candidates_dir,
        output_dir=pairs_dir,
        split="train",
        shard_count=shard_count,
        train_negative_cap_per_s1_source=train_negative_cap,
        test_candidate_cap_per_s1_source=test_candidate_cap,
        state=state,
    )
    build_feature_shards(
        normalized_dir=normalized_dir,
        pairs_dir=pairs_dir,
        output_dir=features_dir,
        cache_dir=cache_dir,
        split="train",
        shard_count=shard_count,
        state=state,
    )
    train_metadata = train_production_model(
        normalized_dir=normalized_dir,
        features_dir=features_dir,
        output_dir=model_dir,
        state=state,
    )
    if not keep_embeddings:
        _clear_embedding_files(cache_dir)

    # Test: run the identical retrieval/features contract, score, and format.
    generate_classical_candidates(
        normalized_dir=normalized_dir,
        output_dir=candidates_dir,
        split="test",
        shard_count=shard_count,
        top_k=classical_top_k,
        state=state,
    )
    generate_ann_candidates(
        normalized_dir=normalized_dir,
        output_dir=candidates_dir,
        cache_dir=cache_dir,
        split="test",
        state=state,
        shard_count=shard_count,
        top_k=ann_top_k,
        encoder_batch_size=encoder_batch_size,
        search_batch_size=search_batch_size,
        index_type="ivf",
        nlist=nlist,
        nprobe=nprobe,
        encoder_device="cuda",
        faiss_device=faiss_device,
        keep_embeddings=True,
    )
    build_pair_shards(
        normalized_dir=normalized_dir,
        candidates_dir=candidates_dir,
        output_dir=pairs_dir,
        split="test",
        shard_count=shard_count,
        train_negative_cap_per_s1_source=train_negative_cap,
        test_candidate_cap_per_s1_source=test_candidate_cap,
        state=state,
    )
    build_feature_shards(
        normalized_dir=normalized_dir,
        pairs_dir=pairs_dir,
        output_dir=features_dir,
        cache_dir=cache_dir,
        split="test",
        shard_count=shard_count,
        state=state,
    )
    submission = score_and_write_submission(
        normalized_dir=normalized_dir,
        features_dir=features_dir,
        model_dir=model_dir,
        output_dir=output_dir,
        state=state,
        shard_count=shard_count,
    )
    if not keep_embeddings:
        _clear_embedding_files(cache_dir)

    result = {
        "status": "complete",
        "preflight": preflight,
        "model": train_metadata,
        "submission": submission,
        "elapsed_hours": round((time.monotonic() - started) / 3600, 3),
    }
    atomic_write_json(work_dir / "pipeline_result.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("/kaggle/working/amazonml_full"))
    parser.add_argument("--shards", type=int, default=100)
    parser.add_argument("--classical-top-k", type=int, default=30)
    parser.add_argument("--ann-top-k", type=int, default=20)
    parser.add_argument("--train-negative-cap", type=int, default=5)
    parser.add_argument("--test-candidate-cap", type=int, default=40)
    parser.add_argument("--encoder-batch-size", type=int, default=512)
    parser.add_argument("--search-batch-size", type=int, default=4096)
    parser.add_argument("--nlist", type=int, default=4096)
    parser.add_argument("--nprobe", type=int, default=64)
    parser.add_argument("--faiss-device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--keep-embeddings", action="store_true")
    args = parser.parse_args()
    dataset_dir = args.dataset_dir or _discover_dataset()
    if dataset_dir is None:
        parser.error(
            "could not uniquely discover the dataset; pass --dataset-dir explicitly"
        )
    run_pipeline(
        dataset_dir=dataset_dir,
        work_dir=args.work_dir,
        shard_count=args.shards,
        classical_top_k=args.classical_top_k,
        ann_top_k=args.ann_top_k,
        train_negative_cap=args.train_negative_cap,
        test_candidate_cap=args.test_candidate_cap,
        encoder_batch_size=args.encoder_batch_size,
        search_batch_size=args.search_batch_size,
        nlist=args.nlist,
        nprobe=args.nprobe,
        faiss_device=args.faiss_device,
        keep_embeddings=args.keep_embeddings,
    )


if __name__ == "__main__":
    main()
