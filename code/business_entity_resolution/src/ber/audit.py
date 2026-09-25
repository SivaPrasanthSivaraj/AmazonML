"""Streaming structural audit for the supplied challenge data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def _scan(path: Path) -> pl.LazyFrame:
    # Polars 1.43's lazy reader treats empty CSV fields as null by default. Keep
    # this wrapper small because `empty_as_null` is only available on read_csv in
    # some Polars releases, not scan_csv.
    return pl.scan_csv(path, separator="\t", encoding="utf8")


def audit_source(path: Path, expected_prefix: str) -> dict[str, object]:
    frame = _scan(path)
    columns = frame.collect_schema().names()
    if columns != SOURCE_COLUMNS:
        raise ValueError(f"{path}: expected {SOURCE_COLUMNS}, found {columns}")

    summary = frame.select(
        pl.len().alias("rows"),
        pl.col("entity_id").n_unique().alias("unique_ids"),
        (~pl.col("entity_id").str.starts_with(expected_prefix))
        .sum()
        .alias("bad_id_prefixes"),
        pl.col("business_name").is_null().sum().alias("missing_names"),
        pl.col("business_address").is_null().sum().alias("missing_addresses"),
        pl.col("country").is_null().sum().alias("missing_countries"),
    ).collect(engine="streaming").to_dicts()[0]
    countries = (
        frame.group_by("country")
        .len(name="rows")
        .sort("country")
        .collect(engine="streaming")
        .to_dicts()
    )
    return {"file": str(path), **summary, "countries": countries}


def audit_ground_truth(path: Path) -> dict[str, object]:
    frame = _scan(path)
    columns = frame.collect_schema().names()
    if columns != GROUND_TRUTH_COLUMNS:
        raise ValueError(f"{path}: expected {GROUND_TRUTH_COLUMNS}, found {columns}")

    frame = frame.with_columns(
        pl.when(pl.col("matched_entity_ids").is_null())
        .then(pl.lit(0))
        .otherwise(pl.col("matched_entity_ids").str.count_matches(",") + 1)
        .alias("match_count")
    )
    summary = frame.select(
        pl.len().alias("rows"),
        pl.col("source1_entity_id").n_unique().alias("unique_s1_ids"),
        (pl.col("match_count") == 0).sum().alias("singletons"),
        pl.col("match_count").sum().alias("links"),
        pl.col("match_count").mean().alias("mean_links_per_s1"),
        pl.col("match_count").max().alias("max_links_per_s1"),
    ).collect(engine="streaming").to_dicts()[0]
    cardinality = (
        frame.group_by("match_count")
        .len(name="s1_entities")
        .sort("match_count")
        .collect(engine="streaming")
        .to_dicts()
    )
    return {"file": str(path), **summary, "cardinality": cardinality}


def run(dataset_dir: Path) -> dict[str, object]:
    train = dataset_dir / "train"
    test = dataset_dir / "test"
    paths = [
        (train / "train_source1.tsv", "S1-"),
        (train / "train_source2.tsv", "S2-"),
        (train / "train_source3.tsv", "S3-"),
        (test / "test_source1.tsv", "S1-"),
        (test / "test_source2.tsv", "S2-"),
        (test / "test_source3.tsv", "S3-"),
    ]
    missing = [str(path) for path, _ in paths if not path.is_file()]
    ground_truth = train / "train_ground_truth.tsv"
    if not ground_truth.is_file():
        missing.append(str(ground_truth))
    if missing:
        raise FileNotFoundError("missing required files: " + ", ".join(missing))
    return {
        "sources": [audit_source(path, prefix) for path, prefix in paths],
        "ground_truth": audit_ground_truth(ground_truth),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset"),
        help="Directory containing train/ and test/ (default: dataset)",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
