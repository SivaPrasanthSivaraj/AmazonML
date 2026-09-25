"""Materialize, deduplicate, and cheaply rank candidates on an S1 subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl
from rapidfuzz import fuzz

from .normalization import normalized_expr
from .qgram_retrieval import _prepared_source, _sample_expr
from .token_retrieval import TOKEN_ROUTES, _document_frequency, _scan


ROUTE_WEIGHTS = {
    "name_tokens": 1.0,
    "address_tokens": 1.0,
    "address_numbers": 1.25,
    "name_char4": 0.20,
    "address_char5": 0.20,
}


def _exact_pairs(
    sample: pl.LazyFrame,
    target: pl.LazyFrame,
    key: str,
    route: str,
) -> pl.DataFrame:
    return (
        sample.filter(pl.col(key) != "")
        .select("s1_id", "country", key)
        .join(
            target.filter(pl.col(key) != "").select("target_id", "country", key),
            on=["country", key],
            how="inner",
        )
        .select(
            "s1_id",
            "target_id",
            "country",
            pl.lit(route).alias("route"),
            pl.lit(5.0).alias("evidence_score"),
            pl.lit(1, dtype=pl.UInt32).alias("shared_keys"),
            pl.lit(0, dtype=pl.UInt32).alias("best_df"),
        )
        .collect(engine="streaming")
    )


def _key_pairs(
    sample: pl.LazyFrame,
    target: pl.LazyFrame,
    eligible_global_df: pl.LazyFrame,
    route: str,
    weight: float,
) -> pl.DataFrame:
    sample_keys = (
        sample.select("s1_id", "country", pl.col(route).alias("token"))
        .explode("token", empty_as_null=True)
        .filter(pl.col("token").is_not_null() & (pl.col("token") != ""))
        .join(eligible_global_df, on=["country", "token"], how="inner")
    )
    target_keys = (
        target.select("target_id", "country", pl.col(route).alias("token"))
        .explode("token", empty_as_null=True)
        .filter(pl.col("token").is_not_null() & (pl.col("token") != ""))
    )
    return (
        sample_keys.join(target_keys, on=["country", "token"], how="inner")
        .with_columns(
            (pl.lit(weight) / pl.col("global_df").cast(pl.Float64).sqrt()).alias(
                "key_score"
            )
        )
        .group_by(["s1_id", "target_id", "country"])
        .agg(
            pl.col("key_score").sum().alias("evidence_score"),
            pl.len().cast(pl.UInt32).alias("shared_keys"),
            pl.col("global_df").min().cast(pl.UInt32).alias("best_df"),
        )
        .with_columns(pl.lit(route).alias("route"))
        .select(
            "s1_id",
            "target_id",
            "country",
            "route",
            "evidence_score",
            "shared_keys",
            "best_df",
        )
        .collect(engine="streaming")
    )


def _truth(train_dir: Path, sample: pl.LazyFrame, source_number: int) -> pl.DataFrame:
    return (
        _scan(train_dir / "train_ground_truth.tsv")
        .filter(pl.col("matched_entity_ids").is_not_null())
        .select(
            pl.col("source1_entity_id").alias("s1_id"),
            pl.col("matched_entity_ids").str.split(",").alias("target_id"),
        )
        .explode("target_id", empty_as_null=True)
        .filter(pl.col("target_id").str.starts_with(f"S{source_number}-"))
        .join(sample.select("s1_id", "country"), on="s1_id", how="inner")
        .collect(engine="streaming")
    )


def _evaluate(
    candidates: pl.DataFrame,
    truth: pl.DataFrame,
    top_ks: tuple[int, ...],
    score_column: str,
) -> dict[str, object]:
    ranked = (
        candidates.sort(
            [
                "target_id",
                score_column,
                "evidence_score",
                "route_count",
                "shared_keys",
                "best_df",
            ],
            descending=[False, True, True, True, True, False],
        )
        .with_columns(
            pl.col("target_id").cum_count().over("target_id").alias("target_rank")
        )
    )
    truth_ranked = truth.join(
        ranked.select("s1_id", "target_id", "target_rank"),
        on=["s1_id", "target_id"],
        how="left",
    )
    recall = []
    for top_k in top_ks:
        rows = (
            truth_ranked.group_by("country")
            .agg(
                pl.len().alias("true_links"),
                (pl.col("target_rank") <= top_k)
                .fill_null(False)
                .sum()
                .alias("retrieved"),
            )
            .sort("country")
            .to_dicts()
        )
        for row in rows:
            row["recall"] = row["retrieved"] / row["true_links"]
        recall.append({"top_k_per_target": top_k, "by_country": rows})

    counts = ranked.group_by("target_id").len(name="candidate_s1_count")
    distribution = counts.select(
        pl.len().alias("targets_with_candidates"),
        pl.col("candidate_s1_count").mean().alias("mean_candidates"),
        pl.col("candidate_s1_count").median().alias("median_candidates"),
        pl.col("candidate_s1_count").quantile(0.95).alias("p95_candidates"),
        pl.col("candidate_s1_count").max().alias("max_candidates"),
    ).to_dicts()[0]
    return {
        "score_column": score_column,
        "unique_candidate_pairs": ranked.height,
        "true_links": truth.height,
        "candidate_recall_before_top_k": truth_ranked["target_rank"].is_not_null().sum()
        / truth.height,
        "candidate_count_by_target": distribution,
        "top_k_recall": recall,
    }


def _add_fuzzy_score(
    candidates: pl.DataFrame,
    train_dir: Path,
    source_number: int,
) -> pl.DataFrame:
    s1_text = _scan(train_dir / "train_source1.tsv").select(
        pl.col("entity_id").alias("s1_id"),
        normalized_expr("business_name", strip_marks=True).alias("s1_name"),
        normalized_expr("business_address", strip_marks=True).alias("s1_address"),
    )
    target_text = _scan(train_dir / f"train_source{source_number}.tsv").select(
        pl.col("entity_id").alias("target_id"),
        normalized_expr("business_name", strip_marks=True).alias("target_name"),
        normalized_expr("business_address", strip_marks=True).alias("target_address"),
    )
    pairs = (
        candidates.lazy()
        .join(s1_text, on="s1_id", how="left")
        .join(target_text, on="target_id", how="left")
        .collect(engine="streaming")
    )
    fuzzy_scores: list[float] = []
    for row in pairs.select(
        "s1_name",
        "target_name",
        "s1_address",
        "target_address",
        "evidence_score",
    ).iter_rows():
        s1_name, target_name, s1_address, target_address, evidence = row
        name_similarity = max(
            fuzz.ratio(s1_name, target_name),
            fuzz.token_set_ratio(s1_name, target_name),
        ) / 100.0
        if s1_address and target_address:
            address_similarity = max(
                fuzz.ratio(s1_address, target_address),
                fuzz.token_set_ratio(s1_address, target_address),
            ) / 100.0
            stronger = max(name_similarity, address_similarity)
            weaker = min(name_similarity, address_similarity)
            text_score = 0.75 * stronger + 0.25 * weaker
        else:
            text_score = name_similarity
        retrieval_score = min(1.0, float(evidence) / 5.0)
        fuzzy_scores.append(0.90 * text_score + 0.10 * retrieval_score)
    return pairs.with_columns(pl.Series("fuzzy_score", fuzzy_scores, dtype=pl.Float32))


def run(
    dataset_dir: Path,
    sample_per_mille: int,
    token_df: int,
    qgram_df: int,
    address_qgram_df: int,
    sources: tuple[int, ...],
    top_ks: tuple[int, ...],
) -> dict[str, object]:
    train_dir = dataset_dir / "train"
    s1 = _prepared_source(train_dir / "train_source1.tsv", "s1_id")
    sample = s1.filter(_sample_expr("s1_id", sample_per_mille))

    thresholds = {route: token_df for route in TOKEN_ROUTES}
    thresholds["name_char4"] = qgram_df
    thresholds["address_char5"] = address_qgram_df
    global_frequencies: dict[str, pl.LazyFrame] = {}
    for route, threshold in thresholds.items():
        # Keep only eligible keys in memory and reuse them for both target sources.
        global_frequencies[route] = (
            _document_frequency(s1, route, "global_df")
            .filter(pl.col("global_df") <= threshold)
            .collect(engine="streaming")
            .lazy()
        )

    source_results = []
    for source_number in sources:
        target_all = _prepared_source(
            train_dir / f"train_source{source_number}.tsv", "target_id"
        )
        truth = _truth(train_dir, sample, source_number)
        # Evaluate sampled holdout targets against the complete S1 candidate
        # universe. Restricting the S1 side to the sample would make top-K ranking
        # unrealistically easy because most competing businesses would disappear.
        target = target_all.join(
            truth.lazy().select("target_id").unique(), on="target_id", how="inner"
        )
        route_frames = [
            _exact_pairs(s1, target, "name_exact_key", "exact_name"),
            _exact_pairs(s1, target, "address_exact_key", "exact_address"),
        ]
        for route, weight in ROUTE_WEIGHTS.items():
            route_frames.append(
                _key_pairs(
                    s1,
                    target,
                    global_frequencies[route],
                    route,
                    weight,
                )
            )
        evidence = pl.concat(route_frames, how="vertical")
        candidates = evidence.group_by(["s1_id", "target_id", "country"]).agg(
            pl.col("evidence_score").sum(),
            pl.col("shared_keys").sum(),
            pl.col("best_df").min(),
            pl.col("route").n_unique().cast(pl.UInt8).alias("route_count"),
        )
        retrieval_result = _evaluate(
            candidates, truth, top_ks, score_column="evidence_score"
        )
        fuzzy_candidates = _add_fuzzy_score(candidates, train_dir, source_number)
        fuzzy_result = _evaluate(
            fuzzy_candidates, truth, top_ks, score_column="fuzzy_score"
        )
        source_results.append(
            {
                "source": f"S{source_number}",
                "route_pair_rows_before_deduplication": evidence.height,
                "retrieval_ranking": retrieval_result,
                "fuzzy_ranking": fuzzy_result,
            }
        )
    return {
        "sample_per_mille": sample_per_mille,
        "token_df": token_df,
        "qgram_df": qgram_df,
        "address_qgram_df": address_qgram_df,
        "ranking_score": "sum(weight / sqrt(S1 document frequency)); exact=5",
        "fuzzy_score": "90% adaptive name/address similarity + 10% retrieval evidence",
        "candidate_universe": "all training S1 records for sampled holdout targets",
        "sources": source_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--sample-per-mille", type=int, default=1)
    parser.add_argument("--token-df", type=int, default=100)
    parser.add_argument("--qgram-df", type=int, default=50)
    parser.add_argument("--address-qgram-df", type=int, default=50)
    parser.add_argument("--sources", type=int, nargs="+", choices=(2, 3), default=[2, 3])
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 2, 3, 5, 10, 20])
    args = parser.parse_args()
    if not 1 <= args.sample_per_mille <= 10:
        parser.error("--sample-per-mille must be between 1 and 10")
    result = run(
        args.dataset_dir,
        args.sample_per_mille,
        args.token_df,
        args.qgram_df,
        args.address_qgram_df,
        tuple(sorted(set(args.sources))),
        tuple(sorted(set(args.top_k))),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
