"""Evaluate rare-token and long-number candidate-generation routes.

The experiment counts candidate *occurrences* without materializing candidate
pairs. A pair sharing two eligible tokens is counted twice, so volume is a safe
upper bound. Recall is exact: each labeled pair is counted at most once per route.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from .normalization import compact_expr, normalized_expr
from .validation import split_expr


DEFAULT_THRESHOLDS = (1, 2, 5, 10, 20, 50, 100)
TOKEN_ROUTES = ("name_tokens", "address_tokens", "address_numbers")
CHARACTER_ROUTES = (
    "name_prefix4",
    "name_suffix4",
    "address_prefix4",
    "address_suffix4",
)
ROUTES = TOKEN_ROUTES + CHARACTER_ROUTES


def _scan(path: Path) -> pl.LazyFrame:
    return pl.scan_csv(path, separator="\t", encoding="utf8")


def _word_tokens(column: str) -> pl.Expr:
    element = pl.element()
    return (
        normalized_expr(column, strip_marks=True)
        .str.split(" ")
        .list.eval(
            element.filter(
                (element.str.len_chars() >= 3)
                & element.str.contains(r"\p{L}")
            )
        )
        .list.unique()
    )


def _long_numbers(column: str) -> pl.Expr:
    # Four or more digits captures ZIP/PIN codes and distinctive premises/phone
    # fragments while avoiding the explosion caused by common 1–2 digit numbers.
    return (
        normalized_expr(column, strip_marks=True)
        .str.extract_all(r"\d{4,}")
        .list.unique()
    )


def _affix_keys(token_column: str, *, suffix: bool) -> pl.Expr:
    element = pl.element()
    sliced = element.str.slice(-4) if suffix else element.str.slice(0, 4)
    return (
        pl.col(token_column)
        .list.eval(sliced.filter(element.str.len_chars() >= 5))
        .list.unique()
    )


def _prepared_source(path: Path, id_name: str) -> pl.LazyFrame:
    return _scan(path).select(
        pl.col("entity_id").alias(id_name),
        pl.col("country"),
        _word_tokens("business_name").alias("name_tokens"),
        _word_tokens("business_address").alias("address_tokens"),
        _long_numbers("business_address").alias("address_numbers"),
        compact_expr("business_name", strip_marks=True).alias("name_exact_key"),
        compact_expr("business_address", strip_marks=True).alias("address_exact_key"),
    ).with_columns(
        _affix_keys("name_tokens", suffix=False).alias("name_prefix4"),
        _affix_keys("name_tokens", suffix=True).alias("name_suffix4"),
        _affix_keys("address_tokens", suffix=False).alias("address_prefix4"),
        _affix_keys("address_tokens", suffix=True).alias("address_suffix4"),
    )


def _document_frequency(
    frame: pl.LazyFrame, route: str, count_name: str
) -> pl.LazyFrame:
    return (
        frame.select("country", pl.col(route).alias("token"))
        .explode("token", empty_as_null=True)
        .filter(pl.col("token").is_not_null() & (pl.col("token") != ""))
        .group_by(["country", "token"])
        .len(name=count_name)
    )


def _candidate_occurrences(
    global_df: pl.LazyFrame,
    holdout_df: pl.LazyFrame,
    target_df: pl.LazyFrame,
    thresholds: tuple[int, ...],
) -> list[dict[str, object]]:
    joined = (
        global_df.join(holdout_df, on=["country", "token"], how="inner")
        .join(target_df, on=["country", "token"], how="inner")
        .with_columns(
            (pl.col("holdout_count") * pl.col("target_count")).alias("occurrences")
        )
    )
    distribution = (
        joined.filter(pl.col("global_df") <= max(thresholds))
        .group_by(["country", "global_df"])
        .agg(
            pl.col("occurrences").sum().alias("occurrences"),
            pl.len().alias("keys"),
        )
        .collect(engine="streaming")
        .to_dicts()
    )
    countries = sorted({row["country"] for row in distribution})
    results: list[dict[str, object]] = []
    for threshold in thresholds:
        rows = []
        for country in countries:
            eligible = [
                row
                for row in distribution
                if row["country"] == country and row["global_df"] <= threshold
            ]
            rows.append(
                {
                    "country": country,
                    "candidate_occurrence_upper_bound": sum(
                        row["occurrences"] for row in eligible
                    ),
                    "blocking_keys": sum(row["keys"] for row in eligible),
                }
            )
        results.append({"max_s1_document_frequency": threshold, "by_country": rows})
    return results


def _minimum_shared_df(
    labeled_pairs: pl.LazyFrame,
    global_df: pl.LazyFrame,
    route: str,
) -> pl.LazyFrame:
    return (
        labeled_pairs.select(
            "s1_id",
            "target_id",
            "country",
            pl.col(route)
            .list.set_intersection(pl.col(f"{route}_target"))
            .alias("token"),
        )
        .explode("token", empty_as_null=True)
        .filter(pl.col("token").is_not_null() & (pl.col("token") != ""))
        .join(global_df, on=["country", "token"], how="inner")
        .group_by(["s1_id", "target_id"])
        .agg(pl.col("global_df").min().alias(f"{route}_min_df"))
    )


def _recall_table(
    labeled_pairs: pl.LazyFrame,
    minima: dict[str, pl.LazyFrame],
    thresholds: tuple[int, ...],
) -> list[dict[str, object]]:
    base = labeled_pairs.select(
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
    base = base.with_columns(
        pl.min_horizontal(
            *[pl.col(f"{route}_min_df").fill_null(large) for route in TOKEN_ROUTES]
        ).alias("token_min_df"),
        pl.min_horizontal(
            *[
                pl.col(f"{route}_min_df").fill_null(large)
                for route in CHARACTER_ROUTES
            ]
        ).alias("character_min_df"),
        pl.min_horizontal(
            *[pl.col(f"{route}_min_df").fill_null(large) for route in ROUTES]
        ).alias("union_min_df")
    )
    evaluated = base.collect(engine="streaming")
    results: list[dict[str, object]] = []
    for threshold in thresholds:
        aggregations: list[pl.Expr] = [pl.len().alias("true_links")]
        for route in ROUTES:
            aggregations.append(
                (pl.col(f"{route}_min_df").fill_null(large) <= threshold)
                .sum()
                .alias(f"{route}_retrieved")
            )
        aggregations.append(
            (pl.col("union_min_df") <= threshold).sum().alias("union_retrieved")
        )
        aggregations.extend(
            [
                (pl.col("token_min_df") <= threshold)
                .sum()
                .alias("whole_token_retrieved"),
                (pl.col("character_min_df") <= threshold)
                .sum()
                .alias("character_retrieved"),
                pl.col("exact_retrieved").sum().alias("exact_retrieved"),
                (
                    (pl.col("union_min_df") <= threshold)
                    & ~pl.col("exact_retrieved")
                )
                .sum()
                .alias("all_key_marginal_links"),
                (
                    (pl.col("character_min_df") <= threshold)
                    & ~(pl.col("token_min_df") <= threshold)
                    & ~pl.col("exact_retrieved")
                )
                .sum()
                .alias("character_marginal_links"),
                (
                    (pl.col("union_min_df") <= threshold)
                    | pl.col("exact_retrieved")
                )
                .sum()
                .alias("combined_retrieved"),
            ]
        )
        rows = (
            evaluated.group_by("country")
            .agg(*aggregations)
            .sort("country")
            .to_dicts()
        )
        for row in rows:
            row["union_recall"] = row["union_retrieved"] / row["true_links"]
            row["exact_recall"] = row["exact_retrieved"] / row["true_links"]
            row["combined_recall"] = row["combined_retrieved"] / row["true_links"]
        results.append({"max_s1_document_frequency": threshold, "by_country": rows})
    return results


def run(dataset_dir: Path, thresholds: tuple[int, ...]) -> dict[str, object]:
    train_dir = dataset_dir / "train"
    s1 = _prepared_source(train_dir / "train_source1.tsv", "s1_id").with_columns(
        split_expr("s1_id").alias("split")
    )
    holdout = s1.filter(pl.col("split") == "holdout")
    ground_truth = (
        _scan(train_dir / "train_ground_truth.tsv")
        .filter(pl.col("matched_entity_ids").is_not_null())
        .select(
            pl.col("source1_entity_id").alias("s1_id"),
            pl.col("matched_entity_ids").str.split(",").alias("target_id"),
        )
        .explode("target_id", empty_as_null=True)
        .join(holdout.select("s1_id"), on="s1_id", how="inner")
    )

    global_frequencies = {
        route: _document_frequency(s1, route, "global_df") for route in ROUTES
    }
    holdout_frequencies = {
        route: _document_frequency(holdout, route, "holdout_count") for route in ROUTES
    }

    output: list[dict[str, object]] = []
    for source_number in (2, 3):
        target = _prepared_source(
            train_dir / f"train_source{source_number}.tsv", "target_id"
        )
        labeled_pairs = (
            ground_truth.filter(
                pl.col("target_id").str.starts_with(f"S{source_number}-")
            )
            .join(holdout, on="s1_id", how="inner")
            .join(target, on="target_id", how="inner", suffix="_target")
        )
        minima = {
            route: _minimum_shared_df(labeled_pairs, global_frequencies[route], route)
            for route in ROUTES
        }
        volumes = {}
        for route in ROUTES:
            target_frequency = _document_frequency(target, route, "target_count")
            volumes[route] = _candidate_occurrences(
                global_frequencies[route],
                holdout_frequencies[route],
                target_frequency,
                thresholds,
            )
        output.append(
            {
                "source": f"S{source_number}",
                "recall": _recall_table(labeled_pairs, minima, thresholds),
                "candidate_occurrence_upper_bounds": volumes,
            }
        )
    return {
        "thresholds": list(thresholds),
        "notes": {
            "candidate_volume": "Upper bound; pairs sharing multiple keys are repeated.",
            "word_token_rule": "accent-folded Unicode tokens with >=3 characters and a letter",
            "number_rule": "address digit sequences with >=4 digits",
            "character_rule": "4-character prefixes/suffixes of words with >=5 characters",
        },
        "sources": output,
    }


def summarize(result: dict[str, object]) -> dict[str, object]:
    """Return the decision-relevant subset of the verbose experiment report."""

    sources = []
    for source in result["sources"]:
        volume_by_threshold: dict[int, int] = {
            threshold: 0 for threshold in result["thresholds"]
        }
        for route_rows in source["candidate_occurrence_upper_bounds"].values():
            for threshold_row in route_rows:
                threshold = threshold_row["max_s1_document_frequency"]
                volume_by_threshold[threshold] += sum(
                    row["candidate_occurrence_upper_bound"]
                    for row in threshold_row["by_country"]
                )
        recall_rows = []
        for threshold_row in source["recall"]:
            recall_rows.append(
                {
                    "max_s1_document_frequency": threshold_row[
                        "max_s1_document_frequency"
                    ],
                    "candidate_occurrence_upper_bound": volume_by_threshold[
                        threshold_row["max_s1_document_frequency"]
                    ],
                    "by_country": [
                        {
                            "country": row["country"],
                            "true_links": row["true_links"],
                            "exact_recall": row["exact_recall"],
                            "whole_token_retrieved": row["whole_token_retrieved"],
                            "character_retrieved": row["character_retrieved"],
                            "character_marginal_links": row[
                                "character_marginal_links"
                            ],
                            "combined_recall": row["combined_recall"],
                        }
                        for row in threshold_row["by_country"]
                    ],
                }
            )
        sources.append({"source": source["source"], "results": recall_rows})
    return {"notes": result["notes"], "sources": sources}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
        help="Maximum country-specific S1 document frequencies to evaluate",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only recall, marginal character gains, and total volume",
    )
    args = parser.parse_args()
    thresholds = tuple(sorted(set(args.thresholds)))
    if not thresholds or thresholds[0] < 1:
        parser.error("thresholds must be positive integers")
    result = run(args.dataset_dir, thresholds)
    if args.summary_only:
        result = summarize(result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
