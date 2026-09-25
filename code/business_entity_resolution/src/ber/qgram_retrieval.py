"""Evaluate internal character q-gram retrieval on a deterministic holdout subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from .normalization import compact_expr
from .token_retrieval import (
    TOKEN_ROUTES,
    _document_frequency,
    _long_numbers,
    _minimum_shared_df,
    _scan,
    _word_tokens,
)


DEFAULT_THRESHOLDS = (1, 2, 5, 10, 20, 50, 100)


def qgrams_expr(column: str, size: int = 4) -> pl.Expr:
    """Generate all overlapping q-grams using vectorized Polars expressions.

    Rust regex does not support look-ahead, so we extract non-overlapping q-grams
    from each possible starting offset. Their union is exactly the overlapping set.
    """

    text = compact_expr(column, strip_marks=True)
    shifted = [text.str.slice(offset).str.extract_all(f".{{{size}}}") for offset in range(size)]
    return pl.concat_list(shifted).list.unique()


def _prepared_source(path: Path, id_name: str) -> pl.LazyFrame:
    return _scan(path).select(
        pl.col("entity_id").alias(id_name),
        pl.col("country"),
        _word_tokens("business_name").alias("name_tokens"),
        _word_tokens("business_address").alias("address_tokens"),
        _long_numbers("business_address").alias("address_numbers"),
        compact_expr("business_name", strip_marks=True).alias("name_exact_key"),
        compact_expr("business_address", strip_marks=True).alias("address_exact_key"),
        qgrams_expr("business_name", size=4).alias("name_char4"),
        qgrams_expr("business_address", size=5).alias("address_char5"),
    )


def _sample_expr(id_column: str, sample_per_mille: int) -> pl.Expr:
    suffix = pl.col(id_column).str.strip_prefix("S1-").cast(pl.UInt64)
    return suffix.mod(1000) >= (1000 - sample_per_mille)


def _candidate_distribution(
    global_df: pl.LazyFrame,
    sample_df: pl.LazyFrame,
    target_df: pl.LazyFrame,
    thresholds: tuple[int, ...],
) -> dict[int, int]:
    distribution = (
        global_df.join(sample_df, on=["country", "token"], how="inner")
        .join(target_df, on=["country", "token"], how="inner")
        .with_columns(
            (pl.col("sample_count") * pl.col("target_count")).alias("occurrences")
        )
        .filter(pl.col("global_df") <= max(thresholds))
        .group_by("global_df")
        .agg(pl.col("occurrences").sum())
        .collect(engine="streaming")
        .to_dicts()
    )
    return {
        threshold: int(
            sum(row["occurrences"] for row in distribution if row["global_df"] <= threshold)
        )
        for threshold in thresholds
    }


def _evaluate_source(
    source_number: int,
    train_dir: Path,
    s1: pl.LazyFrame,
    sample: pl.LazyFrame,
    ground_truth: pl.LazyFrame,
    thresholds: tuple[int, ...],
) -> dict[str, object]:
    target = _prepared_source(
        train_dir / f"train_source{source_number}.tsv", "target_id"
    )
    labeled = (
        ground_truth.filter(pl.col("target_id").str.starts_with(f"S{source_number}-"))
        .join(sample, on="s1_id", how="inner")
        .join(target, on="target_id", how="inner", suffix="_target")
    )

    qgram_routes = ("name_char4", "address_char5")
    global_frequencies = {
        route: _document_frequency(s1, route, "global_df")
        for route in (*TOKEN_ROUTES, *qgram_routes)
    }
    minima = {
        route: _minimum_shared_df(labeled, global_frequencies[route], route)
        for route in (*TOKEN_ROUTES, *qgram_routes)
    }

    base = labeled.select(
        "s1_id",
        "target_id",
        "country",
        (
            (
                (pl.col("name_exact_key") == pl.col("name_exact_key_target"))
                & (pl.col("name_exact_key") != "")
            )
            | (
                (pl.col("address_exact_key") == pl.col("address_exact_key_target"))
                & (pl.col("address_exact_key") != "")
            )
        ).alias("exact_retrieved"),
    )
    for route, route_minima in minima.items():
        base = base.join(route_minima, on=["s1_id", "target_id"], how="left")
    large = 2**31 - 1
    evaluated = base.with_columns(
        pl.min_horizontal(
            *[pl.col(f"{route}_min_df").fill_null(large) for route in TOKEN_ROUTES]
        ).alias("token_min_df")
    ).collect(engine="streaming")

    volumes = {}
    for route in qgram_routes:
        sample_frequency = _document_frequency(sample, route, "sample_count")
        target_frequency = _document_frequency(target, route, "target_count")
        volumes[route] = _candidate_distribution(
            global_frequencies[route], sample_frequency, target_frequency, thresholds
        )

    result_rows = []
    for threshold in thresholds:
        rows = (
            evaluated.group_by("country")
            .agg(
                pl.len().alias("true_links"),
                (
                    pl.col("exact_retrieved") | (pl.col("token_min_df") <= 100)
                ).sum().alias("baseline_retrieved"),
                (pl.col("name_char4_min_df").fill_null(large) <= threshold)
                .sum()
                .alias("name_qgram_retrieved"),
                (pl.col("address_char5_min_df").fill_null(large) <= threshold)
                .sum()
                .alias("address_qgram_retrieved"),
                (
                    (pl.col("name_char4_min_df").fill_null(large) <= threshold)
                    & ~pl.col("exact_retrieved")
                    & ~(pl.col("token_min_df") <= 100)
                )
                .sum()
                .alias("name_qgram_marginal_links"),
                (
                    (pl.col("address_char5_min_df").fill_null(large) <= threshold)
                    & ~(pl.col("name_char4_min_df").fill_null(large) <= threshold)
                    & ~pl.col("exact_retrieved")
                    & ~(pl.col("token_min_df") <= 100)
                )
                .sum()
                .alias("address_qgram_marginal_links"),
                (
                    pl.col("exact_retrieved")
                    | (pl.col("token_min_df") <= 100)
                    | (pl.col("name_char4_min_df").fill_null(large) <= threshold)
                    | (pl.col("address_char5_min_df").fill_null(large) <= threshold)
                )
                .sum()
                .alias("combined_retrieved"),
            )
            .sort("country")
            .to_dicts()
        )
        for row in rows:
            row["baseline_recall"] = row["baseline_retrieved"] / row["true_links"]
            row["combined_recall"] = row["combined_retrieved"] / row["true_links"]
        result_rows.append(
            {
                "max_s1_document_frequency": threshold,
                "candidate_occurrence_upper_bound": sum(
                    volumes[route][threshold] for route in qgram_routes
                ),
                "candidate_occurrences_by_route": {
                    route: volumes[route][threshold] for route in qgram_routes
                },
                "by_country": rows,
            }
        )
    return {"source": f"S{source_number}", "results": result_rows}


def run(
    dataset_dir: Path,
    thresholds: tuple[int, ...],
    sample_per_mille: int,
    sources: tuple[int, ...],
) -> dict[str, object]:
    train_dir = dataset_dir / "train"
    s1 = _prepared_source(train_dir / "train_source1.tsv", "s1_id")
    sample = s1.filter(_sample_expr("s1_id", sample_per_mille))
    sample_counts = (
        sample.group_by("country")
        .len(name="s1_entities")
        .sort("country")
        .collect(engine="streaming")
        .to_dicts()
    )
    ground_truth = (
        _scan(train_dir / "train_ground_truth.tsv")
        .filter(pl.col("matched_entity_ids").is_not_null())
        .select(
            pl.col("source1_entity_id").alias("s1_id"),
            pl.col("matched_entity_ids").str.split(",").alias("target_id"),
        )
        .explode("target_id", empty_as_null=True)
        .join(sample.select("s1_id"), on="s1_id", how="inner")
    )
    return {
        "sample_per_mille": sample_per_mille,
        "sample_counts": sample_counts,
        "baseline_token_df": 100,
        "qgram_size": 4,
        "qgram_sizes": {"name": 4, "address": 5},
        "sources": [
            _evaluate_source(
                source, train_dir, s1, sample, ground_truth, thresholds
            )
            for source in sources
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--thresholds", type=int, nargs="+", default=list(DEFAULT_THRESHOLDS)
    )
    parser.add_argument(
        "--sample-per-mille",
        type=int,
        default=10,
        help="Deterministic S1 sample size; 10 means 1%% of all S1 and is within holdout",
    )
    parser.add_argument("--sources", type=int, nargs="+", choices=(2, 3), default=[2, 3])
    args = parser.parse_args()
    if not 1 <= args.sample_per_mille <= 10:
        parser.error("--sample-per-mille must be between 1 and 10")
    thresholds = tuple(sorted(set(args.thresholds)))
    if not thresholds or thresholds[0] < 1:
        parser.error("thresholds must be positive")
    result = run(
        args.dataset_dir,
        thresholds,
        args.sample_per_mille,
        tuple(sorted(set(args.sources))),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
