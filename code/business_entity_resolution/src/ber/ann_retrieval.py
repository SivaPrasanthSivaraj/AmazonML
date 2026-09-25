"""Benchmark dense nearest-neighbour candidate retrieval on labeled targets.

The expensive S1 embeddings are cached per country and text view.  A small,
deterministic set of labeled S2/S3 targets is queried against the *complete* S1
universe, preserving the competitors that will exist at inference time.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl

from .qgram_retrieval import _sample_expr
from .token_retrieval import _scan


MODEL_NAME = "intfloat/multilingual-e5-small"
MODEL_REVISION = "0e60b8d9d2166d80387f86e3b48ec9ced55f4d15"
EMBEDDING_DIMENSION = 384
VIEWS = ("name", "address", "combined")


def _safe_label(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def _text_for_view(name: str | None, address: str | None, view: str) -> str:
    """Create one E5 symmetric-similarity input without external enrichment."""

    name = (name or "").strip()
    address = (address or "").strip()
    if view == "name":
        body = name
    elif view == "address":
        body = address
    elif view == "combined":
        body = f"business name: {name}; business address: {address}"
    else:
        raise ValueError(f"unknown embedding view: {view}")
    # The E5 model card recommends the query prefix for symmetric similarity.
    return f"query: {body}"


def _texts(frame: pl.DataFrame, view: str) -> list[str]:
    return [
        _text_for_view(name, address, view)
        for name, address in frame.select(
            "business_name", "business_address"
        ).iter_rows()
    ]


def _eligible_queries(frame: pl.DataFrame, view: str) -> pl.DataFrame:
    """Do not spend an address neighbour slot on an absent address."""

    if view != "address":
        return frame
    return frame.filter(
        pl.col("business_address").is_not_null()
        & (pl.col("business_address").str.strip_chars() != "")
    )


def _load_encoder(device: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "ANN dependencies are missing. Install requirements-ann.txt first."
        ) from error
    return SentenceTransformer(
        MODEL_NAME,
        revision=MODEL_REVISION,
        device=None if device == "auto" else device,
    )


def _cache_paths(
    cache_dir: Path, *, role: str, source: str, country: str, view: str
) -> tuple[Path, Path]:
    stem = "__".join(map(_safe_label, (role, source, country, view)))
    return cache_dir / f"{stem}.npy", cache_dir / f"{stem}.json"


def _encode_cached(
    frame: pl.DataFrame,
    *,
    model,
    cache_dir: Path,
    role: str,
    source: str,
    country: str,
    view: str,
    batch_size: int,
) -> np.ndarray:
    """Encode to an on-disk float16 NPY, safely resuming only complete caches."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    array_path, metadata_path = _cache_paths(
        cache_dir, role=role, source=source, country=country, view=view
    )
    expected = {
        "model": MODEL_NAME,
        "revision": MODEL_REVISION,
        "rows": frame.height,
        "dimension": EMBEDDING_DIMENSION,
        "dtype": "float16",
        "view": view,
    }
    if array_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata == expected:
            cached = np.load(array_path, mmap_mode="r")
            if cached.shape == (frame.height, EMBEDDING_DIMENSION):
                print(f"cache hit: {array_path}", flush=True)
                return cached

    partial_path = array_path.with_suffix(".partial.npy")
    embeddings = np.lib.format.open_memmap(
        partial_path,
        mode="w+",
        dtype=np.float16,
        shape=(frame.height, EMBEDDING_DIMENSION),
    )
    offset = 0
    started = time.monotonic()
    for batch in frame.iter_slices(n_rows=batch_size):
        values = model.encode(
            _texts(batch, view),
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        if values.shape[1] != EMBEDDING_DIMENSION:
            raise RuntimeError(
                f"expected {EMBEDDING_DIMENSION} dimensions, got {values.shape[1]}"
            )
        embeddings[offset : offset + batch.height] = values.astype(np.float16)
        offset += batch.height
        if offset == frame.height or offset % (batch_size * 100) == 0:
            elapsed = max(time.monotonic() - started, 1e-6)
            print(
                f"encoded {source}/{country}/{view}: {offset:,}/{frame.height:,} "
                f"({offset / elapsed:,.0f} rows/s)",
                flush=True,
            )
    embeddings.flush()
    del embeddings
    partial_path.replace(array_path)
    metadata_path.write_text(
        json.dumps(expected, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return np.load(array_path, mmap_mode="r")


def _add_in_batches(index, embeddings: np.ndarray, batch_rows: int = 100_000) -> None:
    for start in range(0, len(embeddings), batch_rows):
        index.add(np.asarray(embeddings[start : start + batch_rows], dtype=np.float32))


def _build_index(
    embeddings: np.ndarray,
    *,
    index_type: str,
    nlist: int,
    nprobe: int,
    faiss_device: str,
    seed: int,
):
    try:
        import faiss
    except ImportError as error:
        raise RuntimeError(
            "FAISS is missing. Install requirements-ann.txt or Kaggle's FAISS package."
        ) from error

    dimension = embeddings.shape[1]
    if index_type == "flat":
        index = faiss.IndexFlatIP(dimension)
    else:
        effective_nlist = min(nlist, max(1, len(embeddings) // 40))
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(
            quantizer, dimension, effective_nlist, faiss.METRIC_INNER_PRODUCT
        )
        rng = np.random.default_rng(seed)
        training_count = min(len(embeddings), max(100_000, effective_nlist * 40))
        training_rows = rng.choice(len(embeddings), training_count, replace=False)
        index.train(np.asarray(embeddings[training_rows], dtype=np.float32))
        index.nprobe = min(nprobe, effective_nlist)

    _add_in_batches(index, embeddings)
    wants_gpu = faiss_device == "gpu" or (
        faiss_device == "auto" and hasattr(faiss, "StandardGpuResources")
    )
    if wants_gpu:
        if not hasattr(faiss, "StandardGpuResources"):
            if faiss_device == "gpu":
                raise RuntimeError("the installed FAISS build has no GPU support")
        else:
            resources = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(resources, 0, index)
            # Keep the resource alive for as long as the GPU index.
            index._gpu_resources = resources
    return index


def _search(index, queries: np.ndarray, top_k: int, batch_rows: int = 4096):
    score_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    for start in range(0, len(queries), batch_rows):
        scores, indices = index.search(
            np.asarray(queries[start : start + batch_rows], dtype=np.float32), top_k
        )
        score_parts.append(scores)
        index_parts.append(indices)
    return np.vstack(score_parts), np.vstack(index_parts)


def _torch_search(
    database: np.ndarray,
    queries: np.ndarray,
    top_k: int,
    batch_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact normalized-inner-product search on CUDA without GPU FAISS."""

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for --search-backend torch") from error
    if not torch.cuda.is_available():
        raise RuntimeError("--search-backend torch requires a CUDA GPU")
    database_gpu = torch.as_tensor(
        np.asarray(database), dtype=torch.float16, device="cuda"
    )
    score_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(queries), batch_rows):
            query_gpu = torch.as_tensor(
                np.asarray(queries[start : start + batch_rows]),
                dtype=torch.float16,
                device="cuda",
            )
            scores, indices = torch.topk(
                query_gpu @ database_gpu.T, k=top_k, dim=1, sorted=True
            )
            score_parts.append(scores.float().cpu().numpy())
            index_parts.append(indices.cpu().numpy())
            del query_gpu, scores, indices
    del database_gpu
    return np.vstack(score_parts), np.vstack(index_parts)


def _resolve_search_backend(requested: str, index_type: str) -> str:
    if index_type == "ivf":
        if requested == "torch":
            raise ValueError("IVF search requires the FAISS backend")
        return "faiss"
    if requested != "auto":
        return requested
    try:
        import faiss

        if hasattr(faiss, "StandardGpuResources"):
            return "faiss"
    except ImportError:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            return "torch"
    except ImportError:
        pass
    return "faiss"


def _candidate_frame(
    *,
    target_ids: list[str],
    s1_ids: list[str],
    scores: np.ndarray,
    indices: np.ndarray,
    source: str,
    country: str,
    view: str,
) -> pl.DataFrame:
    width = indices.shape[1]
    target_column = np.repeat(np.asarray(target_ids, dtype=object), width)
    flat_indices = indices.reshape(-1)
    valid = flat_indices >= 0
    return pl.DataFrame(
        {
            "target_id": target_column[valid],
            "s1_id": np.asarray(s1_ids, dtype=object)[flat_indices[valid]],
            "source": source,
            "country": country,
            "view": view,
            "ann_rank": np.tile(np.arange(1, width + 1), len(target_ids))[valid],
            "ann_score": scores.reshape(-1)[valid],
        }
    ).with_columns(
        pl.col("ann_rank").cast(pl.UInt16),
        pl.col("ann_score").cast(pl.Float32),
    )


def _recall_rows(
    truth: pl.DataFrame,
    candidates: pl.DataFrame,
    top_ks: Iterable[int],
) -> list[dict[str, object]]:
    """Return per-country link recall for candidates carrying view/rank columns."""

    results: list[dict[str, object]] = []
    for top_k in top_ks:
        eligible = candidates.filter(pl.col("ann_rank") <= top_k)
        union_pairs = eligible.select("s1_id", "target_id").unique()
        matched = truth.join(
            union_pairs.with_columns(pl.lit(True).alias("retrieved")),
            on=["s1_id", "target_id"],
            how="left",
        )
        by_country = (
            matched.group_by("country")
            .agg(
                pl.len().alias("true_links"),
                pl.col("retrieved").fill_null(False).sum().alias("retrieved"),
            )
            .sort("country")
            .to_dicts()
        )
        for row in by_country:
            row["recall"] = row["retrieved"] / row["true_links"]
        total = sum(row["true_links"] for row in by_country)
        retrieved = sum(row["retrieved"] for row in by_country)
        counts = union_pairs.group_by("target_id").len(name="candidate_count")
        results.append(
            {
                "top_k_per_view": top_k,
                "retrieved": retrieved,
                "true_links": total,
                "recall": retrieved / total,
                "unique_candidate_pairs": union_pairs.height,
                "mean_candidates_per_retrieved_target": (
                    counts["candidate_count"].mean() if counts.height else 0.0
                ),
                "by_country": by_country,
            }
        )
    return results


def _marginal_summary(
    truth: pl.DataFrame,
    ann_candidates: pl.DataFrame,
    classical_candidates: pl.DataFrame,
    top_k: int,
) -> dict[str, object]:
    ann_pairs = (
        ann_candidates.filter(pl.col("ann_rank") <= top_k)
        .select("s1_id", "target_id")
        .unique()
    )
    classical_pairs = classical_candidates.select("s1_id", "target_id").unique()
    flags = (
        truth.join(
            classical_pairs.with_columns(pl.lit(True).alias("classical")),
            on=["s1_id", "target_id"],
            how="left",
        )
        .join(
            ann_pairs.with_columns(pl.lit(True).alias("ann")),
            on=["s1_id", "target_id"],
            how="left",
        )
        .with_columns(
            pl.col("classical").fill_null(False),
            pl.col("ann").fill_null(False),
        )
    )
    total = flags.height
    classical = flags["classical"].sum()
    ann = flags["ann"].sum()
    overlap = flags.select((pl.col("classical") & pl.col("ann")).sum()).item()
    union = flags.select((pl.col("classical") | pl.col("ann")).sum()).item()
    return {
        "ann_top_k_per_view": top_k,
        "true_links": total,
        "classical_retrieved": classical,
        "ann_retrieved": ann,
        "overlap": overlap,
        "ann_marginal_links": union - classical,
        "union_retrieved": union,
        "classical_recall": classical / total,
        "ann_recall": ann / total,
        "union_recall": union / total,
    }


def _load_s1(train_dir: Path) -> pl.DataFrame:
    return (
        _scan(train_dir / "train_source1.tsv")
        .select("entity_id", "country", "business_name", "business_address")
        .rename({"entity_id": "s1_id"})
        .collect(engine="streaming")
    )


def _load_truth(train_dir: Path, sample_per_mille: int, source_number: int) -> pl.DataFrame:
    sampled_s1 = (
        _scan(train_dir / "train_source1.tsv")
        .select(pl.col("entity_id").alias("s1_id"), "country")
        .filter(_sample_expr("s1_id", sample_per_mille))
    )
    return (
        _scan(train_dir / "train_ground_truth.tsv")
        .filter(pl.col("matched_entity_ids").is_not_null())
        .select(
            pl.col("source1_entity_id").alias("s1_id"),
            pl.col("matched_entity_ids").str.split(",").alias("target_id"),
        )
        .explode("target_id", empty_as_null=True)
        .filter(pl.col("target_id").str.starts_with(f"S{source_number}-"))
        .join(sampled_s1, on="s1_id", how="inner")
        .collect(engine="streaming")
    )


def _load_targets(train_dir: Path, source_number: int, truth: pl.DataFrame) -> pl.DataFrame:
    return (
        _scan(train_dir / f"train_source{source_number}.tsv")
        .select("entity_id", "business_name", "business_address")
        .rename({"entity_id": "target_id"})
        .join(truth.lazy().select("target_id", "country"), on="target_id", how="inner")
        .collect(engine="streaming")
    )


def run(
    *,
    dataset_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    sample_per_mille: int,
    sources: tuple[int, ...],
    views: tuple[str, ...],
    top_ks: tuple[int, ...],
    batch_size: int,
    encoder_device: str,
    faiss_device: str,
    search_backend: str,
    search_batch_size: int,
    index_type: str,
    nlist: int,
    nprobe: int,
    classical_candidates_dir: Path | None,
    seed: int,
) -> dict[str, object]:
    train_dir = dataset_dir / "train"
    output_dir.mkdir(parents=True, exist_ok=True)
    model = _load_encoder(encoder_device)
    resolved_search_backend = _resolve_search_backend(search_backend, index_type)
    print(f"search backend: {resolved_search_backend}", flush=True)
    s1 = _load_s1(train_dir)
    source_summaries = []

    for source_number in sources:
        source = f"S{source_number}"
        truth = _load_truth(train_dir, sample_per_mille, source_number)
        targets = _load_targets(train_dir, source_number, truth)
        candidate_parts: list[pl.DataFrame] = []
        for country in sorted(truth["country"].unique().to_list()):
            s1_country = s1.filter(pl.col("country") == country)
            target_country = targets.filter(pl.col("country") == country)
            for view in views:
                target_view = _eligible_queries(target_country, view)
                print(
                    f"ANN {source}/{country}/{view}: {target_view.height:,} targets "
                    f"against {s1_country.height:,} S1 records",
                    flush=True,
                )
                database_embeddings = _encode_cached(
                    s1_country,
                    model=model,
                    cache_dir=cache_dir,
                    role="database",
                    source="S1",
                    country=country,
                    view=view,
                    batch_size=batch_size,
                )
                query_embeddings = _encode_cached(
                    target_view,
                    model=model,
                    cache_dir=cache_dir,
                    role="query",
                    source=source,
                    country=country,
                    view=view,
                    batch_size=batch_size,
                )
                if resolved_search_backend == "torch":
                    scores, indices = _torch_search(
                        database_embeddings,
                        query_embeddings,
                        max(top_ks),
                        search_batch_size,
                    )
                    index = None
                else:
                    index = _build_index(
                        database_embeddings,
                        index_type=index_type,
                        nlist=nlist,
                        nprobe=nprobe,
                        faiss_device=faiss_device,
                        seed=seed,
                    )
                    scores, indices = _search(
                        index, query_embeddings, max(top_ks), search_batch_size
                    )
                candidate_parts.append(
                    _candidate_frame(
                        target_ids=target_view["target_id"].to_list(),
                        s1_ids=s1_country["s1_id"].to_list(),
                        scores=scores,
                        indices=indices,
                        source=source,
                        country=country,
                        view=view,
                    )
                )
                del index, database_embeddings, query_embeddings, scores, indices

        candidates = pl.concat(candidate_parts, how="vertical")
        candidates_path = output_dir / f"ann_candidates_{source}.parquet"
        candidates.write_parquet(candidates_path, compression="zstd")
        per_view = {
            view: _recall_rows(
                truth, candidates.filter(pl.col("view") == view), top_ks
            )
            for view in views
        }
        union_recall = _recall_rows(truth, candidates, top_ks)
        source_summary: dict[str, object] = {
            "source": source,
            "true_links": truth.height,
            "candidate_file": str(candidates_path),
            "per_view": per_view,
            "union_across_views": union_recall,
        }
        if classical_candidates_dir is not None:
            classical_path = classical_candidates_dir / f"classical_candidates_{source}.parquet"
            if not classical_path.exists():
                raise FileNotFoundError(
                    f"missing {classical_path}; rerun candidate_ranking with --output-dir"
                )
            classical = pl.read_parquet(classical_path)
            source_summary["marginal_over_classical"] = [
                _marginal_summary(truth, candidates, classical, top_k)
                for top_k in top_ks
            ]
        source_summaries.append(source_summary)

    result = {
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "sample_per_mille": sample_per_mille,
        "views": views,
        "index": {
            "type": index_type,
            "metric": "cosine via normalized inner product",
            "nlist": nlist if index_type == "ivf" else None,
            "nprobe": nprobe if index_type == "ivf" else None,
            "faiss_device": faiss_device,
            "search_backend": resolved_search_backend,
        },
        "sources": source_summaries,
    }
    summary_path = output_dir / "ann_summary.json"
    summary_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/ann"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/ann_cache"))
    parser.add_argument("--sample-per-mille", type=int, default=1)
    parser.add_argument("--sources", type=int, nargs="+", choices=(2, 3), default=[2, 3])
    parser.add_argument("--views", nargs="+", choices=VIEWS, default=list(VIEWS))
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 5, 10, 20, 50])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--encoder-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--faiss-device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument(
        "--search-backend", choices=("auto", "faiss", "torch"), default="auto"
    )
    parser.add_argument("--search-batch-size", type=int, default=128)
    parser.add_argument("--index-type", choices=("flat", "ivf"), default="flat")
    parser.add_argument("--nlist", type=int, default=4096)
    parser.add_argument("--nprobe", type=int, default=64)
    parser.add_argument("--classical-candidates-dir", type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if not 1 <= args.sample_per_mille <= 10:
        parser.error("--sample-per-mille must be between 1 and 10")
    top_ks = tuple(sorted(set(args.top_k)))
    if not top_ks or top_ks[0] < 1:
        parser.error("--top-k values must be positive")
    if min(args.batch_size, args.search_batch_size, args.nlist, args.nprobe) < 1:
        parser.error("batch sizes, nlist, and nprobe must be positive")
    result = run(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        sample_per_mille=args.sample_per_mille,
        sources=tuple(sorted(set(args.sources))),
        views=tuple(dict.fromkeys(args.views)),
        top_ks=top_ks,
        batch_size=args.batch_size,
        encoder_device=args.encoder_device,
        faiss_device=args.faiss_device,
        search_backend=args.search_backend,
        search_batch_size=args.search_batch_size,
        index_type=args.index_type,
        nlist=args.nlist,
        nprobe=args.nprobe,
        classical_candidates_dir=args.classical_candidates_dir,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
