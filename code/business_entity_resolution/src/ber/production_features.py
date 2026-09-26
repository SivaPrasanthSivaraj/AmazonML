"""Leakage-safe lexical and direct-E5 features for production pair shards."""

from __future__ import annotations

import gc
import json
import re
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz

from .ann_retrieval import VIEWS, _cache_paths, _eligible_queries
from .pipeline_state import PipelineState, atomic_write_json, signature
from .production_data import source_output_path
from .production_pairs import pair_output_path


SAFE_BASE_FEATURES = (
    "name_ratio",
    "name_wratio",
    "name_token_set",
    "name_length_ratio",
    "name_exact",
    "address_ratio",
    "address_wratio",
    "address_token_set",
    "address_length_ratio",
    "address_digit_jaccard",
    "address_exact",
    "address_missing",
    "source_is_s3",
    "country_is_us",
    "country_is_india",
    "e5_name_cosine",
    "e5_address_cosine",
    "e5_combined_cosine",
    "e5_best_cosine",
    "e5_combined_minus_name",
    "e5_combined_minus_address",
    "e5_name_address_product",
)


_DIGIT_RE = re.compile(r"\d+")


def feature_output_path(root: Path, split: str, source: int, shard: int) -> Path:
    return root / "features" / split / f"S{source}" / f"part-{shard:04d}.parquet"


def _length_ratio(left: str, right: str) -> float:
    longer = max(len(left), len(right))
    return min(len(left), len(right)) / longer if longer else 1.0


def _digit_jaccard(left: str, right: str) -> float:
    left_digits = set(_DIGIT_RE.findall(left))
    right_digits = set(_DIGIT_RE.findall(right))
    union = left_digits | right_digits
    return len(left_digits & right_digits) / len(union) if union else 0.0


def _lexical_features(frame: pl.DataFrame) -> pl.DataFrame:
    columns: dict[str, list[float | int]] = {
        name: [] for name in SAFE_BASE_FEATURES[:12]
    }
    for s1_name, target_name, s1_address, target_address in frame.select(
        "s1_name", "target_name", "s1_address", "target_address"
    ).iter_rows():
        s1_name = s1_name or ""
        target_name = target_name or ""
        s1_address = s1_address or ""
        target_address = target_address or ""
        columns["name_ratio"].append(fuzz.ratio(s1_name, target_name) / 100.0)
        columns["name_wratio"].append(fuzz.WRatio(s1_name, target_name) / 100.0)
        columns["name_token_set"].append(
            fuzz.token_set_ratio(s1_name, target_name) / 100.0
        )
        columns["name_length_ratio"].append(_length_ratio(s1_name, target_name))
        columns["name_exact"].append(int(bool(s1_name) and s1_name == target_name))
        address_missing = not s1_address or not target_address
        columns["address_ratio"].append(
            0.0 if address_missing else fuzz.ratio(s1_address, target_address) / 100.0
        )
        columns["address_wratio"].append(
            0.0 if address_missing else fuzz.WRatio(s1_address, target_address) / 100.0
        )
        columns["address_token_set"].append(
            0.0
            if address_missing
            else fuzz.token_set_ratio(s1_address, target_address) / 100.0
        )
        columns["address_length_ratio"].append(
            0.0 if address_missing else _length_ratio(s1_address, target_address)
        )
        columns["address_digit_jaccard"].append(
            0.0 if address_missing else _digit_jaccard(s1_address, target_address)
        )
        columns["address_exact"].append(
            int(not address_missing and s1_address == target_address)
        )
        columns["address_missing"].append(int(address_missing))
    return frame.with_columns(
        *[
            pl.Series(name, values, dtype=pl.Float32)
            for name, values in columns.items()
        ],
        (pl.col("source") == "S3").cast(pl.Float32).alias("source_is_s3"),
        (pl.col("country") == "US").cast(pl.Float32).alias("country_is_us"),
        (pl.col("country") == "India").cast(pl.Float32).alias("country_is_india"),
    )


def _embedding_rows(frame: pl.DataFrame, view: str) -> pl.DataFrame:
    return _eligible_queries(frame, view).with_row_index("embedding_row")


def _direct_e5_scores(
    pairs: pl.DataFrame,
    *,
    s1_all: pl.DataFrame,
    target_all: pl.DataFrame,
    cache_dir: Path,
    source: int,
    view: str,
    dot_batch_size: int,
) -> np.ndarray:
    result = np.zeros(pairs.height, dtype=np.float32)
    for country in pairs["country"].unique().to_list():
        pair_country = pairs.filter(pl.col("country") == country)
        s1_rows = _embedding_rows(
            s1_all.filter(pl.col("country") == country), view
        ).select(pl.col("entity_id").alias("s1_id"), "embedding_row")
        target_rows = _embedding_rows(
            target_all.filter(pl.col("country") == country), view
        ).select(pl.col("entity_id").alias("target_id"), "embedding_row")
        indexed = (
            pair_country.select("_pair_row", "s1_id", "target_id")
            .join(s1_rows.rename({"embedding_row": "s1_embedding_row"}), on="s1_id", how="left")
            .join(
                target_rows.rename({"embedding_row": "target_embedding_row"}),
                on="target_id",
                how="left",
            )
        )
        valid = indexed.filter(
            pl.col("s1_embedding_row").is_not_null()
            & pl.col("target_embedding_row").is_not_null()
        )
        if valid.is_empty():
            continue
        s1_array_path, _ = _cache_paths(
            cache_dir,
            role="query",
            source="S1",
            country=country,
            view=view,
        )
        target_array_path, _ = _cache_paths(
            cache_dir,
            role="database",
            source=f"S{source}",
            country=country,
            view=view,
        )
        if not s1_array_path.exists() or not target_array_path.exists():
            raise FileNotFoundError(
                "E5 embedding cache is missing. Run ANN generation with "
                "keep_embeddings=True before the feature stage. Missing "
                f"{s1_array_path if not s1_array_path.exists() else target_array_path}"
            )
        s1_embeddings = np.load(s1_array_path, mmap_mode="r")
        target_embeddings = np.load(target_array_path, mmap_mode="r")
        pair_rows = valid["_pair_row"].to_numpy().astype(np.int64)
        s1_indices = valid["s1_embedding_row"].to_numpy().astype(np.int64)
        target_indices = valid["target_embedding_row"].to_numpy().astype(np.int64)
        for start in range(0, valid.height, dot_batch_size):
            end = min(start + dot_batch_size, valid.height)
            scores = np.einsum(
                "ij,ij->i",
                np.asarray(s1_embeddings[s1_indices[start:end]], dtype=np.float32),
                np.asarray(
                    target_embeddings[target_indices[start:end]], dtype=np.float32
                ),
                optimize=True,
            )
            result[pair_rows[start:end]] = scores
        del s1_embeddings, target_embeddings
    return result


def _atomic_write(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.unlink(missing_ok=True)
    frame.write_parquet(partial, compression="zstd")
    partial.replace(path)


def build_feature_shards(
    *,
    normalized_dir: Path,
    pairs_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    split: str,
    state: PipelineState,
    shard_count: int,
    views: tuple[str, ...] = VIEWS,
    dot_batch_size: int = 50_000,
) -> dict[str, object]:
    missing_views = set(VIEWS) - set(views)
    if missing_views:
        raise ValueError(
            "the validated model requires all E5 views; missing "
            f"{sorted(missing_views)}"
        )
    stage = f"features_{split}"
    stage_signature = signature(
        {
            "stage": stage,
            "features": SAFE_BASE_FEATURES,
            "views": views,
            "shard_count": shard_count,
            "version": 1,
        }
    )
    manifest = output_dir / "features" / split / "manifest.json"
    if state.stage_complete(stage, stage_signature, (manifest,)):
        print(f"[{stage}] checkpoint complete; skipping", flush=True)
        return json.loads(manifest.read_text(encoding="utf-8"))
    state.begin_stage(stage, stage_signature)

    s1_all = pl.read_parquet(source_output_path(normalized_dir, split, 1)).select(
        "entity_id", "country", "business_name", "business_address", "name_norm", "address_norm"
    )
    counts: dict[str, dict[str, int]] = {}
    for source in (2, 3):
        target_all = pl.read_parquet(
            source_output_path(normalized_dir, split, source)
        ).select(
            "entity_id",
            "country",
            "business_name",
            "business_address",
            "name_norm",
            "address_norm",
        )
        s1_text = s1_all.select(
            pl.col("entity_id").alias("s1_id"),
            pl.col("name_norm").alias("s1_name"),
            pl.col("address_norm").alias("s1_address"),
        )
        target_text = target_all.select(
            pl.col("entity_id").alias("target_id"),
            pl.col("name_norm").alias("target_name"),
            pl.col("address_norm").alias("target_address"),
        )
        for shard in range(shard_count):
            output = feature_output_path(output_dir, split, source, shard)
            shard_name = f"S{source}_{shard:04d}"
            shard_signature = signature(
                {"stage": stage_signature, "source": source, "shard": shard}
            )
            if state.shard_complete(stage, shard_name, shard_signature, (output,)):
                saved = pl.read_parquet(output, columns=["s1_id"])
                counts[shard_name] = {"rows": saved.height}
                continue
            print(
                f"[{stage}] source=S{source} shard={shard + 1}/{shard_count}",
                flush=True,
            )
            pairs = pl.read_parquet(pair_output_path(pairs_dir, split, source, shard))
            pairs = (
                pairs.join(s1_text, on="s1_id", how="left")
                .join(target_text, on="target_id", how="left")
                .with_row_index("_pair_row")
            )
            if pairs.select(
                pl.any_horizontal(
                    pl.col("s1_name").is_null(), pl.col("target_name").is_null()
                )
            ).item():
                raise RuntimeError(f"missing joined names in {shard_name}")
            pairs = _lexical_features(pairs)
            for view in VIEWS:
                print(f"[{stage}] direct E5 {shard_name}/{view}", flush=True)
                scores = _direct_e5_scores(
                    pairs,
                    s1_all=s1_all,
                    target_all=target_all,
                    cache_dir=cache_dir,
                    source=source,
                    view=view,
                    dot_batch_size=dot_batch_size,
                )
                pairs = pairs.with_columns(
                    pl.Series(f"e5_{view}_cosine", scores, dtype=pl.Float32)
                )
            pairs = pairs.with_columns(
                pl.max_horizontal(
                    "e5_name_cosine", "e5_address_cosine", "e5_combined_cosine"
                ).alias("e5_best_cosine"),
                (pl.col("e5_combined_cosine") - pl.col("e5_name_cosine")).alias(
                    "e5_combined_minus_name"
                ),
                (pl.col("e5_combined_cosine") - pl.col("e5_address_cosine")).alias(
                    "e5_combined_minus_address"
                ),
                (pl.col("e5_name_cosine") * pl.col("e5_address_cosine")).alias(
                    "e5_name_address_product"
                ),
            )
            identity_columns = ["s1_id", "target_id", "source", "country"]
            if split == "train":
                identity_columns += ["validation_split", "is_match"]
            saved = pairs.select(*identity_columns, *SAFE_BASE_FEATURES)
            _atomic_write(saved, output)
            counts[shard_name] = {
                "rows": saved.height,
                "positives": int(saved["is_match"].sum()) if split == "train" else 0,
            }
            state.complete_shard(
                stage,
                shard_name,
                shard_signature,
                (output,),
                counts[shard_name],
            )
            del pairs, saved
            gc.collect()
        del target_all
        gc.collect()

    result = {
        "split": split,
        "features": list(SAFE_BASE_FEATURES),
        "shard_count": shard_count,
        "rows": sum(value["rows"] for value in counts.values()),
        "positives": sum(value.get("positives", 0) for value in counts.values()),
        "shards": counts,
    }
    atomic_write_json(manifest, result)
    state.complete_stage(stage, (manifest,))
    return result
