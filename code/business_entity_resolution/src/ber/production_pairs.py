"""Union production retrieval routes into bounded train/test pair shards."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from .pipeline_state import PipelineState, atomic_write_json, signature
from .production_data import source_output_path, truth_output_path


def pair_output_path(root: Path, split: str, source: int, shard: int) -> Path:
    return root / "pairs" / split / f"S{source}" / f"part-{shard:04d}.parquet"


def _numeric_shard(column: str, shard: int, shard_count: int) -> pl.Expr:
    return (
        pl.col(column)
        .str.strip_prefix("S1-")
        .cast(pl.UInt64)
        .mod(shard_count)
        == shard
    )


def _read_ann_shard(
    candidates_dir: Path, split: str, source: int, shard: int
) -> pl.DataFrame:
    paths = sorted(
        (candidates_dir / "ann" / split / f"S{source}").glob(
            f"*/*/part-{shard:04d}.parquet"
        )
    )
    if not paths:
        raise FileNotFoundError(
            f"no ANN files for {split}/S{source}/shard {shard:04d}"
        )
    rows = pl.concat([pl.read_parquet(path) for path in paths], how="vertical")
    return rows.group_by(["s1_id", "target_id", "source", "country"]).agg(
        pl.col("ann_score").max().alias("retrieval_ann_best_score"),
        pl.col("ann_rank").min().alias("retrieval_ann_best_rank"),
        pl.col("ann_view").n_unique().cast(pl.UInt8).alias("retrieval_ann_view_count"),
        pl.col("ann_score")
        .filter(pl.col("ann_view") == "name")
        .max()
        .alias("retrieval_ann_name_score"),
        pl.col("ann_score")
        .filter(pl.col("ann_view") == "address")
        .max()
        .alias("retrieval_ann_address_score"),
        pl.col("ann_score")
        .filter(pl.col("ann_view") == "combined")
        .max()
        .alias("retrieval_ann_combined_score"),
    )


def _read_classical_shard(
    candidates_dir: Path, split: str, source: int, shard: int
) -> pl.DataFrame:
    path = (
        candidates_dir
        / "classical"
        / split
        / f"S{source}"
        / f"part-{shard:04d}.parquet"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    return pl.read_parquet(path)


def _truth_shard(
    normalized_dir: Path, source: int, shard: int, shard_count: int
) -> pl.DataFrame:
    return (
        pl.scan_parquet(truth_output_path(normalized_dir))
        .filter(
            (pl.col("source") == f"S{source}")
            & _numeric_shard("s1_id", shard, shard_count)
        )
        .select("s1_id", "target_id", "source")
        .collect(engine="streaming")
    )


def _atomic_write(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.unlink(missing_ok=True)
    frame.write_parquet(partial, compression="zstd")
    partial.replace(path)


def build_pair_shards(
    *,
    normalized_dir: Path,
    candidates_dir: Path,
    output_dir: Path,
    split: str,
    state: PipelineState,
    shard_count: int,
    train_negative_cap_per_s1_source: int = 5,
    test_candidate_cap_per_s1_source: int = 30,
) -> dict[str, object]:
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    cap = (
        train_negative_cap_per_s1_source
        if split == "train"
        else test_candidate_cap_per_s1_source
    )
    if cap < 1:
        raise ValueError("pair cap must be positive")
    stage = f"pair_union_{split}"
    stage_signature = signature(
        {
            "stage": stage,
            "shard_count": shard_count,
            "cap": cap,
            "train_truth_union": split == "train",
            "version": 1,
        }
    )
    manifest = output_dir / "pairs" / split / "manifest.json"
    if state.stage_complete(stage, stage_signature, (manifest,)):
        print(f"[{stage}] checkpoint complete; skipping", flush=True)
        return json.loads(manifest.read_text(encoding="utf-8"))

    state.begin_stage(stage, stage_signature)
    s1_columns = [
        pl.col("entity_id").alias("s1_id"),
        "country",
    ]
    if split == "train":
        s1_columns.append(pl.col("split").alias("validation_split"))
    s1 = pl.read_parquet(source_output_path(normalized_dir, split, 1)).select(
        s1_columns
    )
    counts: dict[str, dict[str, int]] = {}

    for source in (2, 3):
        for shard in range(shard_count):
            output = pair_output_path(output_dir, split, source, shard)
            shard_name = f"S{source}_{shard:04d}"
            shard_signature = signature(
                {"stage": stage_signature, "source": source, "shard": shard}
            )
            if state.shard_complete(stage, shard_name, shard_signature, (output,)):
                frame = pl.read_parquet(output)
                counts[shard_name] = {
                    "rows": frame.height,
                    "positives": int(frame["is_match"].sum()) if split == "train" else 0,
                }
                continue

            print(
                f"[{stage}] source=S{source} shard={shard + 1}/{shard_count}",
                flush=True,
            )
            classical = _read_classical_shard(
                candidates_dir, split, source, shard
            )
            ann = _read_ann_shard(candidates_dir, split, source, shard)
            pairs = classical.join(
                ann,
                on=["s1_id", "target_id", "source", "country"],
                how="full",
                coalesce=True,
            )
            pairs = pairs.with_columns(
                pl.max_horizontal(
                    pl.col("retrieval_ann_best_score").fill_null(-1.0),
                    (
                        pl.col("classical_evidence").fill_null(0.0) / 5.0
                    ).clip(0.0, 1.0),
                ).alias("retrieval_priority")
            )

            if split == "train":
                truth = _truth_shard(
                    normalized_dir, source, shard, shard_count
                ).with_columns(pl.lit(True).alias("is_match"))
                pairs = pairs.join(
                    truth,
                    on=["s1_id", "target_id", "source"],
                    how="full",
                    coalesce=True,
                ).with_columns(pl.col("is_match").fill_null(False))
                positives = pairs.filter(pl.col("is_match"))
                negatives = (
                    pairs.filter(~pl.col("is_match"))
                    .sort(
                        ["s1_id", "retrieval_priority", "target_id"],
                        descending=[False, True, False],
                    )
                    .with_columns(
                        pl.col("s1_id").cum_count().over("s1_id").alias("negative_rank")
                    )
                    .filter(pl.col("negative_rank") <= cap)
                    .drop("negative_rank")
                )
                pairs = pl.concat([positives, negatives], how="diagonal_relaxed")
            else:
                pairs = (
                    pairs.sort(
                        ["s1_id", "retrieval_priority", "target_id"],
                        descending=[False, True, False],
                    )
                    .with_columns(
                        pl.col("s1_id").cum_count().over("s1_id").alias("candidate_rank")
                    )
                    .filter(pl.col("candidate_rank") <= cap)
                    .drop("candidate_rank")
                )

            pairs = pairs.drop("country").join(s1, on="s1_id", how="left")
            if pairs["country"].null_count():
                raise RuntimeError(f"missing S1 country in {shard_name}")
            pairs = pairs.sort(["s1_id", "source", "target_id"])
            _atomic_write(pairs, output)
            positives_count = (
                int(pairs["is_match"].sum()) if split == "train" else 0
            )
            counts[shard_name] = {
                "rows": pairs.height,
                "positives": positives_count,
            }
            state.complete_shard(
                stage,
                shard_name,
                shard_signature,
                (output,),
                counts[shard_name],
            )

    result = {
        "split": split,
        "shard_count": shard_count,
        "cap_per_s1_source": cap,
        "rows": sum(value["rows"] for value in counts.values()),
        "positives": sum(value["positives"] for value in counts.values()),
        "shards": counts,
    }
    atomic_write_json(manifest, result)
    state.complete_stage(stage, (manifest,))
    return result
