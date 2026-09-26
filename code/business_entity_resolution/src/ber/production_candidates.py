"""Sharded production classical candidate generation.

This module runs in the submission direction: every S1 record queries the full
S2/S3 target pools.  Candidate shards are bounded per S1 and independently
checkpointed, allowing a long full-data run to resume safely.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import polars as pl

from .pipeline_state import PipelineState, atomic_write_json, signature
from .production_data import source_output_path


ROUTE_CONFIG = {
    "exact_name": {"column": "name_compact", "weight": 5.0, "max_s1_df": 100, "max_target_df": 500},
    "exact_address": {"column": "address_compact", "weight": 5.0, "max_s1_df": 100, "max_target_df": 500},
    "name_tokens": {"column": "name_norm", "weight": 1.0, "max_s1_df": 100, "max_target_df": 2_000},
    "address_tokens": {"column": "address_norm", "weight": 1.0, "max_s1_df": 100, "max_target_df": 2_000},
    "address_numbers": {"column": "address_norm", "weight": 1.25, "max_s1_df": 100, "max_target_df": 2_000},
    "name_char4": {"column": "name_compact", "weight": 0.20, "max_s1_df": 50, "max_target_df": 1_000},
    "address_char5": {"column": "address_compact", "weight": 0.20, "max_s1_df": 50, "max_target_df": 1_000},
}


def _tokens(column: str) -> pl.Expr:
    element = pl.element()
    return (
        pl.col(column)
        .fill_null("")
        .str.split(" ")
        .list.eval(
            element.filter(
                (element.str.len_chars() >= 3) & element.str.contains(r"\p{L}")
            )
        )
        .list.unique()
    )


def _numbers(column: str) -> pl.Expr:
    return pl.col(column).fill_null("").str.extract_all(r"\d{4,}").list.unique()


def _qgrams(column: str, size: int) -> pl.Expr:
    text = pl.col(column).fill_null("")
    return pl.concat_list(
        [text.str.slice(offset).str.extract_all(f".{{{size}}}") for offset in range(size)]
    ).list.unique()


def route_keys(route: str) -> pl.Expr:
    config = ROUTE_CONFIG[route]
    column = config["column"]
    if route.startswith("exact_"):
        return pl.concat_list(pl.col(column).fill_null(""))
    if route in {"name_tokens", "address_tokens"}:
        return _tokens(column)
    if route == "address_numbers":
        return _numbers(column)
    if route == "name_char4":
        return _qgrams(column, 4)
    if route == "address_char5":
        return _qgrams(column, 5)
    raise ValueError(f"unknown route: {route}")


def _exploded_keys(path: Path, id_column: str, route: str) -> pl.LazyFrame:
    return (
        pl.scan_parquet(path)
        .select(
            pl.col("entity_id").alias(id_column),
            "country",
            route_keys(route).alias("key"),
        )
        .explode("key", empty_as_null=True)
        .filter(pl.col("key").is_not_null() & (pl.col("key") != ""))
        .unique(subset=[id_column, "country", "key"])
    )


def _atomic_sink(frame: pl.LazyFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.unlink(missing_ok=True)
    frame.sink_parquet(partial, compression="zstd")
    partial.replace(path)


def _index_path(root: Path, split: str, source: int, route: str) -> Path:
    return root / "indexes" / split / f"S{source}" / f"{route}.parquet"


def _candidate_path(root: Path, split: str, source: int, shard: int) -> Path:
    return root / "classical" / split / f"S{source}" / f"part-{shard:04d}.parquet"


def _shard_filter(column: str, shard: int, shard_count: int) -> pl.Expr:
    numeric = pl.col(column).str.strip_prefix("S1-").cast(pl.UInt64)
    return numeric.mod(shard_count) == shard


def build_classical_indexes(
    *,
    normalized_dir: Path,
    output_dir: Path,
    split: str,
    state: PipelineState,
) -> dict[str, int]:
    stage = f"classical_indexes_{split}"
    stage_signature = signature(
        {"stage": stage, "routes": ROUTE_CONFIG, "version": 1}
    )
    outputs = tuple(
        _index_path(output_dir, split, source, route)
        for source in (2, 3)
        for route in ROUTE_CONFIG
    )
    if state.stage_complete(stage, stage_signature, outputs):
        print(f"[{stage}] checkpoint complete; skipping", flush=True)
        return {str(path): int(pl.scan_parquet(path).select(pl.len()).collect().item()) for path in outputs}

    state.begin_stage(stage, stage_signature)
    s1_path = source_output_path(normalized_dir, split, 1)
    counts: dict[str, int] = {}
    for route, config in ROUTE_CONFIG.items():
        print(f"[{stage}] S1 document frequencies: {route}", flush=True)
        s1_df = (
            _exploded_keys(s1_path, "s1_id", route)
            .group_by(["country", "key"])
            .len(name="s1_df")
            .filter(pl.col("s1_df") <= int(config["max_s1_df"]))
            .collect(engine="streaming")
            .lazy()
        )
        for source in (2, 3):
            output = _index_path(output_dir, split, source, route)
            shard_name = f"S{source}_{route}"
            shard_signature = signature(
                {"stage": stage_signature, "source": source, "route": route}
            )
            if not state.shard_complete(
                stage, shard_name, shard_signature, (output,)
            ):
                target_path = source_output_path(normalized_dir, split, source)
                print(f"[{stage}] building {shard_name}", flush=True)
                eligible_target = (
                    _exploded_keys(target_path, "target_id", route)
                    .join(s1_df, on=["country", "key"], how="inner")
                    .with_columns(
                        pl.len().over(["country", "key"]).alias("target_df")
                    )
                    .filter(pl.col("target_df") <= int(config["max_target_df"]))
                    .select("target_id", "country", "key", "s1_df", "target_df")
                )
                _atomic_sink(eligible_target, output)
                rows = int(pl.scan_parquet(output).select(pl.len()).collect().item())
                state.complete_shard(
                    stage, shard_name, shard_signature, (output,), {"rows": rows}
                )
            counts[str(output)] = int(
                pl.scan_parquet(output).select(pl.len()).collect().item()
            )
    state.complete_stage(stage, outputs)
    return counts


def generate_classical_candidates(
    *,
    normalized_dir: Path,
    output_dir: Path,
    split: str,
    shard_count: int,
    top_k: int,
    state: PipelineState,
) -> dict[str, object]:
    if shard_count < 1 or top_k < 1:
        raise ValueError("shard_count and top_k must be positive")
    build_classical_indexes(
        normalized_dir=normalized_dir,
        output_dir=output_dir,
        split=split,
        state=state,
    )
    stage = f"classical_candidates_{split}"
    stage_signature = signature(
        {
            "stage": stage,
            "routes": ROUTE_CONFIG,
            "shard_count": shard_count,
            "top_k": top_k,
            "version": 1,
        }
    )
    manifest = output_dir / "classical" / split / "manifest.json"
    all_outputs = tuple(
        _candidate_path(output_dir, split, source, shard)
        for source in (2, 3)
        for shard in range(shard_count)
    )
    if state.stage_complete(stage, stage_signature, (manifest,)) and all(
        path.exists() for path in all_outputs
    ):
        print(f"[{stage}] checkpoint complete; skipping", flush=True)
        return json.loads(manifest.read_text(encoding="utf-8"))

    state.begin_stage(stage, stage_signature)
    s1_path = source_output_path(normalized_dir, split, 1)
    counts: dict[str, int] = {}
    for source in (2, 3):
        for shard in range(shard_count):
            output = _candidate_path(output_dir, split, source, shard)
            shard_name = f"S{source}_{shard:04d}"
            shard_signature = signature(
                {"stage": stage_signature, "source": source, "shard": shard}
            )
            if state.shard_complete(stage, shard_name, shard_signature, (output,)):
                counts[shard_name] = int(
                    pl.scan_parquet(output).select(pl.len()).collect().item()
                )
                continue
            print(
                f"[{stage}] source=S{source} shard={shard + 1}/{shard_count}",
                flush=True,
            )
            s1_shard = pl.scan_parquet(s1_path).filter(
                _shard_filter("entity_id", shard, shard_count)
            )
            route_parts: list[pl.LazyFrame] = []
            for route, config in ROUTE_CONFIG.items():
                s1_keys = (
                    s1_shard.select(
                        pl.col("entity_id").alias("s1_id"),
                        "country",
                        route_keys(route).alias("key"),
                    )
                    .explode("key", empty_as_null=True)
                    .filter(pl.col("key").is_not_null() & (pl.col("key") != ""))
                    .unique(subset=["s1_id", "country", "key"])
                )
                target_index = pl.scan_parquet(
                    _index_path(output_dir, split, source, route)
                )
                route_parts.append(
                    s1_keys.join(
                        target_index, on=["country", "key"], how="inner"
                    ).select(
                        "s1_id",
                        "target_id",
                        "country",
                        pl.lit(f"S{source}").alias("source"),
                        pl.lit(route).alias("route"),
                        (
                            pl.lit(float(config["weight"]))
                            / pl.col("s1_df").cast(pl.Float64).sqrt()
                        ).alias("route_score"),
                        pl.col("s1_df").alias("key_df"),
                    )
                )
            evidence = pl.concat(route_parts, how="vertical")
            ranked = (
                evidence.group_by(["s1_id", "target_id", "country", "source"])
                .agg(
                    pl.col("route_score").sum().alias("classical_evidence"),
                    pl.col("route").n_unique().cast(pl.UInt8).alias("classical_route_count"),
                    pl.len().cast(pl.UInt16).alias("classical_shared_keys"),
                    pl.col("key_df").min().cast(pl.UInt32).alias("classical_best_df"),
                )
                .sort(
                    ["s1_id", "classical_evidence", "classical_route_count", "classical_shared_keys", "classical_best_df", "target_id"],
                    descending=[False, True, True, True, False, False],
                )
                .with_columns(
                    pl.col("s1_id").cum_count().over("s1_id").alias("classical_rank")
                )
                .filter(pl.col("classical_rank") <= top_k)
            )
            _atomic_sink(ranked, output)
            rows = int(pl.scan_parquet(output).select(pl.len()).collect().item())
            counts[shard_name] = rows
            state.complete_shard(
                stage, shard_name, shard_signature, (output,), {"rows": rows}
            )

    result = {
        "split": split,
        "shard_count": shard_count,
        "top_k_per_s1_source": top_k,
        "routes": ROUTE_CONFIG,
        "rows_by_shard": counts,
        "total_rows": sum(counts.values()),
    }
    atomic_write_json(manifest, result)
    state.complete_stage(stage, (manifest,))
    return result
