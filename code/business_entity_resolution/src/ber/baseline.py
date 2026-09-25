"""Measure deterministic splits and exact normalized candidate routes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from .normalization import compact_expr
from .validation import split_expr


def _scan(path: Path) -> pl.LazyFrame:
    return pl.scan_csv(path, separator="\t", encoding="utf8")


def _prepared_source(path: Path, id_name: str) -> pl.LazyFrame:
    return _scan(path).select(
        pl.col("entity_id").alias(id_name),
        pl.col("country"),
        compact_expr("business_name").alias("name_key"),
        compact_expr("business_name", strip_marks=True).alias("name_folded_key"),
        compact_expr("business_address").alias("address_key"),
        compact_expr("business_address", strip_marks=True).alias("address_folded_key"),
    )


def _candidate_volume(
    s1: pl.LazyFrame, target: pl.LazyFrame, keys: list[str]
) -> dict[str, int]:
    """Count exact-join pairs without materializing the pair table."""

    left = s1.filter(pl.all_horizontal(pl.col(k) != "" for k in keys)).group_by(
        ["country", *keys]
    ).len(name="left_count")
    right = target.filter(pl.all_horizontal(pl.col(k) != "" for k in keys)).group_by(
        ["country", *keys]
    ).len(name="right_count")
    rows = (
        left.join(right, on=["country", *keys], how="inner")
        .with_columns(
            (pl.col("left_count") * pl.col("right_count")).alias("pairs")
        )
        .group_by("country")
        .agg(pl.col("pairs").sum())
        .collect(engine="streaming")
        .to_dicts()
    )
    return {row["country"]: int(row["pairs"] or 0) for row in rows}


def run(dataset_dir: Path) -> dict[str, object]:
    train_dir = dataset_dir / "train"
    s1 = _prepared_source(train_dir / "train_source1.tsv", "s1_id").with_columns(
        split_expr("s1_id").alias("split")
    )
    split_counts = (
        s1.group_by(["split", "country"])
        .len(name="s1_entities")
        .sort(["split", "country"])
        .collect(engine="streaming")
        .to_dicts()
    )
    holdout = s1.filter(pl.col("split") == "holdout")

    ground_truth = (
        _scan(train_dir / "train_ground_truth.tsv")
        .filter(pl.col("matched_entity_ids").is_not_null())
        .select(
            pl.col("source1_entity_id").alias("s1_id"),
            pl.col("matched_entity_ids").str.split(",").alias("target_id"),
        )
        # Preserve the Polars 1.x behavior explicitly for 2.x compatibility.
        .explode("target_id", empty_as_null=True)
    )
    holdout_links = ground_truth.join(
        holdout.select("s1_id"), on="s1_id", how="inner"
    )

    route_results: list[dict[str, object]] = []
    for source_number in (2, 3):
        target = _prepared_source(
            train_dir / f"train_source{source_number}.tsv", "target_id"
        )
        labeled_pairs = (
            holdout_links.filter(
                pl.col("target_id").str.starts_with(f"S{source_number}-")
            )
            .join(holdout, on="s1_id", how="inner", suffix="_s1")
            .join(target, on="target_id", how="inner", suffix="_target")
            .with_columns(
                (
                    (pl.col("name_key") == pl.col("name_key_target"))
                    & (pl.col("name_key") != "")
                ).alias("name_exact"),
                (
                    (pl.col("name_folded_key") == pl.col("name_folded_key_target"))
                    & (pl.col("name_folded_key") != "")
                ).alias("name_folded_exact"),
                (
                    (pl.col("address_key") == pl.col("address_key_target"))
                    & (pl.col("address_key") != "")
                ).alias("address_exact"),
                (
                    (
                        pl.col("address_folded_key")
                        == pl.col("address_folded_key_target")
                    )
                    & (pl.col("address_folded_key") != "")
                ).alias("address_folded_exact"),
            )
        )
        stats = labeled_pairs.select(
            pl.len().alias("true_links"),
            pl.col("name_exact").sum(),
            pl.col("name_folded_exact").sum(),
            pl.col("address_exact").sum(),
            pl.col("address_folded_exact").sum(),
            pl.any_horizontal(
                "name_exact",
                "name_folded_exact",
                "address_exact",
                "address_folded_exact",
            ).sum().alias("union_retrieved_links"),
        ).collect(engine="streaming").to_dicts()[0]
        country_stats = (
            labeled_pairs.group_by("country")
            .agg(
                pl.len().alias("true_links"),
                pl.col("name_exact").sum(),
                pl.col("name_folded_exact").sum(),
                pl.col("address_exact").sum(),
                pl.col("address_folded_exact").sum(),
                pl.any_horizontal(
                    "name_exact",
                    "name_folded_exact",
                    "address_exact",
                    "address_folded_exact",
                ).sum().alias("union_retrieved_links"),
            )
            .sort("country")
            .collect(engine="streaming")
            .to_dicts()
        )
        for row in country_stats:
            row["union_recall"] = row["union_retrieved_links"] / row["true_links"]

        # Inclusion/exclusion over exact name and address routes. Folded variants
        # are reported for recall but omitted here because their four-way overlap
        # requires materializing a larger union; later candidate stages write and
        # deduplicate the actual pair table.
        name_pairs = _candidate_volume(holdout, target, ["name_key"])
        address_pairs = _candidate_volume(holdout, target, ["address_key"])
        both_pairs = _candidate_volume(holdout, target, ["name_key", "address_key"])
        countries = set(name_pairs) | set(address_pairs) | set(both_pairs)
        pair_volume_by_country = {
            country: name_pairs.get(country, 0)
            + address_pairs.get(country, 0)
            - both_pairs.get(country, 0)
            for country in sorted(countries)
        }
        stats.update(
            source=f"S{source_number}",
            exact_candidate_pairs=sum(pair_volume_by_country.values()),
            exact_name_pairs=sum(name_pairs.values()),
            exact_address_pairs=sum(address_pairs.values()),
            exact_candidate_pairs_by_country=pair_volume_by_country,
            recall_by_country=country_stats,
        )
        stats["union_recall"] = (
            stats["union_retrieved_links"] / stats["true_links"]
            if stats["true_links"]
            else 0.0
        )
        route_results.append(stats)

    return {"split_counts": split_counts, "exact_routes": route_results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
