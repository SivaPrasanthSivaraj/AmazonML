"""Streaming preparation of challenge TSV files for production stages."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from .normalization import compact_expr, normalized_expr
from .pipeline_state import PipelineState, signature
from .validation import split_expr


SOURCE_COLUMNS = (
    "entity_id",
    "country",
    "business_name",
    "business_address",
)


def required_dataset_files(dataset_dir: Path) -> tuple[Path, ...]:
    return (
        *(dataset_dir / "train" / f"train_source{number}.tsv" for number in (1, 2, 3)),
        dataset_dir / "train" / "train_ground_truth.tsv",
        *(dataset_dir / "test" / f"test_source{number}.tsv" for number in (1, 2, 3)),
    )


def validate_dataset(dataset_dir: Path) -> None:
    missing = [str(path) for path in required_dataset_files(dataset_dir) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"challenge dataset is incomplete; missing: {missing}")


def source_output_path(output_dir: Path, split: str, source_number: int) -> Path:
    return output_dir / f"{split}_source{source_number}.parquet"


def truth_output_path(output_dir: Path) -> Path:
    return output_dir / "train_truth_links.parquet"


def _source_lazy(path: Path, split: str, source_number: int) -> pl.LazyFrame:
    frame = pl.scan_csv(path, separator="\t", encoding="utf8").select(
        *SOURCE_COLUMNS
    )
    prepared = frame.select(
        pl.col("entity_id"),
        pl.col("country").cast(pl.Utf8),
        pl.col("business_name").cast(pl.Utf8).fill_null(""),
        pl.col("business_address").cast(pl.Utf8).fill_null(""),
        normalized_expr("business_name", strip_marks=True).alias("name_norm"),
        normalized_expr("business_address", strip_marks=True).alias("address_norm"),
        compact_expr("business_name", strip_marks=True).alias("name_compact"),
        compact_expr("business_address", strip_marks=True).alias("address_compact"),
        pl.lit(f"S{source_number}").alias("source"),
    )
    if source_number == 1 and split == "train":
        prepared = prepared.with_columns(split_expr("entity_id").alias("split"))
    return prepared


def _truth_lazy(path: Path) -> pl.LazyFrame:
    return (
        pl.scan_csv(path, separator="\t", encoding="utf8")
        .select(
            pl.col("source1_entity_id").alias("s1_id"),
            pl.col("matched_entity_ids")
            .fill_null("")
            .str.split(",")
            .alias("target_id"),
        )
        .explode("target_id", empty_as_null=True)
        .filter(pl.col("target_id").is_not_null() & (pl.col("target_id") != ""))
        .with_columns(
            pl.when(pl.col("target_id").str.starts_with("S2-"))
            .then(pl.lit("S2"))
            .when(pl.col("target_id").str.starts_with("S3-"))
            .then(pl.lit("S3"))
            .otherwise(pl.lit(None, dtype=pl.Utf8))
            .alias("source")
        )
        .filter(pl.col("source").is_not_null())
        .unique(subset=["s1_id", "target_id"])
    )


def _atomic_sink(frame: pl.LazyFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.unlink(missing_ok=True)
    frame.sink_parquet(partial, compression="zstd")
    partial.replace(path)


def prepare_production_data(
    dataset_dir: Path,
    output_dir: Path,
    state: PipelineState,
) -> dict[str, object]:
    """Normalize all challenge sources and truth with per-file resume checks."""

    validate_dataset(dataset_dir)
    stage = "prepare_data"
    input_stats = {
        str(path.relative_to(dataset_dir)): {
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in required_dataset_files(dataset_dir)
    }
    stage_signature = signature(
        {"stage": stage, "normalization_version": 1, "inputs": input_stats}
    )
    outputs = tuple(
        source_output_path(output_dir, split, source)
        for split in ("train", "test")
        for source in (1, 2, 3)
    ) + (truth_output_path(output_dir),)
    if state.stage_complete(stage, stage_signature, outputs):
        print("[prepare_data] checkpoint complete; skipping", flush=True)
        return {"skipped": True, "outputs": [str(path) for path in outputs]}

    state.begin_stage(stage, stage_signature)
    row_counts: dict[str, int] = {}
    for split in ("train", "test"):
        for source in (1, 2, 3):
            output = source_output_path(output_dir, split, source)
            shard = f"{split}_source{source}"
            shard_signature = signature(
                {"stage": stage_signature, "shard": shard}
            )
            if state.shard_complete(stage, shard, shard_signature, (output,)):
                print(f"[prepare_data] checkpoint {shard}; skipping", flush=True)
            else:
                source_path = dataset_dir / split / f"{split}_source{source}.tsv"
                print(f"[prepare_data] normalizing {source_path}", flush=True)
                _atomic_sink(_source_lazy(source_path, split, source), output)
                rows = pl.scan_parquet(output).select(pl.len()).collect().item()
                state.complete_shard(
                    stage, shard, shard_signature, (output,), {"rows": rows}
                )
            row_counts[shard] = int(
                pl.scan_parquet(output).select(pl.len()).collect().item()
            )

    truth_output = truth_output_path(output_dir)
    truth_shard_signature = signature(
        {"stage": stage_signature, "shard": "train_truth_links"}
    )
    if not state.shard_complete(
        stage, "train_truth_links", truth_shard_signature, (truth_output,)
    ):
        print("[prepare_data] exploding training truth", flush=True)
        _atomic_sink(
            _truth_lazy(dataset_dir / "train" / "train_ground_truth.tsv"),
            truth_output,
        )
        truth_rows = pl.scan_parquet(truth_output).select(pl.len()).collect().item()
        state.complete_shard(
            stage,
            "train_truth_links",
            truth_shard_signature,
            (truth_output,),
            {"rows": truth_rows},
        )
    row_counts["train_truth_links"] = int(
        pl.scan_parquet(truth_output).select(pl.len()).collect().item()
    )
    state.complete_stage(stage, outputs)
    return {"skipped": False, "rows": row_counts, "outputs": [str(p) for p in outputs]}
