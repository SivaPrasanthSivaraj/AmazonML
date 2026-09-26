"""Score test shards and write challenge-valid matching/candidate TSV files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from .pipeline_state import PipelineState, atomic_write_json, signature
from .production_data import source_output_path
from .production_features import SAFE_BASE_FEATURES, feature_output_path


def _atomic_write_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.unlink(missing_ok=True)
    frame.write_parquet(partial, compression="zstd")
    partial.replace(path)


def _write_grouped_shard(
    *,
    s1: pl.DataFrame,
    pairs: pl.DataFrame,
    value_column: str,
    output: Path,
) -> None:
    grouped = (
        pairs.group_by("s1_id")
        .agg(pl.col("target_id").unique().sort().str.join(",").alias(value_column))
    )
    complete = (
        s1.join(grouped, on="s1_id", how="left")
        .with_columns(pl.col(value_column).fill_null(""))
        .rename({"s1_id": "source1_entity_id"})
        .sort("source1_entity_id")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    complete.write_csv(output, separator="\t", include_header=False)


    output.write_text(
        output.read_text(encoding='utf-8').replace(chr(34) + chr(34), ''),
        encoding='utf-8',
    )


def _concatenate_tsv_parts(
    paths: list[Path], output: Path, header: str
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    with partial.open("w", encoding="utf-8", newline="") as destination:
        destination.write(header + "\n")
        for path in paths:
            with path.open("r", encoding="utf-8") as source:
                for line in source:
                    destination.write(line)
    partial.replace(output)


def score_and_write_submission(
    *,
    normalized_dir: Path,
    features_dir: Path,
    model_dir: Path,
    output_dir: Path,
    state: PipelineState,
    shard_count: int,
    enforce_target_exclusivity: bool = True,
) -> dict[str, object]:
    try:
        import lightgbm as lgb
    except ImportError as error:
        raise RuntimeError("install LightGBM before test scoring") from error

    model_path = model_dir / "model" / "final_master_model.txt"
    metadata_path = model_dir / "model" / "master_metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("trained model artifacts are missing")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    threshold = float(metadata["threshold"])
    stage = "test_scoring_and_submission"
    stage_signature = signature(
        {
            "stage": stage,
            "model_bytes": model_path.stat().st_size,
            "threshold": threshold,
            "features": SAFE_BASE_FEATURES,
            "shard_count": shard_count,
            "target_exclusivity": enforce_target_exclusivity,
            "version": 1,
        }
    )
    matching_path = output_dir / "matching_results.tsv"
    candidate_path = output_dir / "candidate_pairs.tsv"
    summary_path = output_dir / "submission_summary.json"
    outputs = (matching_path, candidate_path, summary_path)
    if state.stage_complete(stage, stage_signature, outputs):
        print(f"[{stage}] checkpoint complete; skipping", flush=True)
        return json.loads(summary_path.read_text(encoding="utf-8"))
    state.begin_stage(stage, stage_signature)

    booster = lgb.Booster(model_file=str(model_path))
    scored_dir = output_dir / "scored_parts"
    scored_paths: list[Path] = []
    for source in (2, 3):
        for shard in range(shard_count):
            features = pl.read_parquet(
                feature_output_path(features_dir, "test", source, shard)
            )
            scores = booster.predict(features.select(SAFE_BASE_FEATURES).to_numpy())
            scored = features.select("s1_id", "target_id", "source").with_columns(
                pl.Series("score", np.asarray(scores, dtype=np.float32))
            )
            path = scored_dir / f"S{source}" / f"part-{shard:04d}.parquet"
            _atomic_write_parquet(scored, path)
            scored_paths.append(path)
            print(
                f"[{stage}] scored S{source} shard {shard + 1}/{shard_count}",
                flush=True,
            )

    scored_scan = pl.scan_parquet(scored_paths)
    predicted = scored_scan.filter(pl.col("score") >= threshold)
    if enforce_target_exclusivity:
        predicted = (
            predicted.sort(
                ["target_id", "score", "s1_id"],
                descending=[False, True, False],
            )
            .unique(subset=["target_id"], keep="first", maintain_order=True)
        )
    selected_path = scored_dir / "selected_matches.parquet"
    partial_selected = selected_path.with_suffix(".parquet.partial")
    partial_selected.unlink(missing_ok=True)
    predicted.sink_parquet(partial_selected, compression="zstd")
    partial_selected.replace(selected_path)

    s1_all = pl.read_parquet(source_output_path(normalized_dir, "test", 1)).select(
        pl.col("entity_id").alias("s1_id")
    )
    selected = pl.read_parquet(selected_path)
    candidate_parts: list[Path] = []
    matching_parts: list[Path] = []
    for shard in range(shard_count):
        s1_shard = s1_all.filter(
            pl.col("s1_id")
            .str.strip_prefix("S1-")
            .cast(pl.UInt64)
            .mod(shard_count)
            == shard
        )
        shard_ids = s1_shard["s1_id"]
        candidate_rows = (
            scored_scan.filter(pl.col("s1_id").is_in(shard_ids.implode()))
            .select("s1_id", "target_id")
            .collect(engine="streaming")
        )
        matching_rows = selected.filter(
            pl.col("s1_id").is_in(shard_ids.implode())
        ).select("s1_id", "target_id")
        candidate_part = output_dir / "tsv_parts" / "candidates" / f"part-{shard:04d}.tsv"
        matching_part = output_dir / "tsv_parts" / "matches" / f"part-{shard:04d}.tsv"
        _write_grouped_shard(
            s1=s1_shard,
            pairs=candidate_rows,
            value_column="candidate_entity_ids",
            output=candidate_part,
        )
        _write_grouped_shard(
            s1=s1_shard,
            pairs=matching_rows,
            value_column="matched_entity_ids",
            output=matching_part,
        )
        candidate_parts.append(candidate_part)
        matching_parts.append(matching_part)

    _concatenate_tsv_parts(
        matching_parts,
        matching_path,
        "source1_entity_id\tmatched_entity_ids",
    )
    _concatenate_tsv_parts(
        candidate_parts,
        candidate_path,
        "source1_entity_id\tcandidate_entity_ids",
    )
    result = {
        "threshold": threshold,
        "target_exclusivity": enforce_target_exclusivity,
        "s1_entities": s1_all.height,
        "candidate_pairs": int(scored_scan.select(pl.len()).collect().item()),
        "predicted_links": selected.height,
        "matching_results": str(matching_path),
        "candidate_pairs_file": str(candidate_path),
    }
    atomic_write_json(summary_path, result)
    state.complete_stage(stage, outputs)
    return result
