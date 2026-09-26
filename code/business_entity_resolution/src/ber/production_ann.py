"""Sharded S1-to-target ANN retrieval for train and test production runs."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import polars as pl

from .ann_retrieval import (
    EMBEDDING_DIMENSION,
    MODEL_NAME,
    MODEL_REVISION,
    VIEWS,
    _build_index,
    _cache_paths,
    _eligible_queries,
    _encode_cached,
    _load_encoder,
    _search,
)
from .pipeline_state import PipelineState, atomic_write_json, signature
from .production_data import source_output_path


def _safe_label(value: str) -> str:
    return "".join(
        character.lower() if character.isalnum() else "_" for character in value
    ).strip("_")


def _shard_mask(ids: list[str], shard: int, shard_count: int) -> np.ndarray:
    return np.fromiter(
        (int(value.removeprefix("S1-")) % shard_count == shard for value in ids),
        dtype=bool,
        count=len(ids),
    )


def _ann_output_path(
    root: Path,
    split: str,
    source: int,
    view: str,
    country: str,
    shard: int,
) -> Path:
    return (
        root
        / "ann"
        / split
        / f"S{source}"
        / view
        / _safe_label(country)
        / f"part-{shard:04d}.parquet"
    )


def _atomic_write_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.unlink(missing_ok=True)
    frame.write_parquet(partial, compression="zstd")
    partial.replace(path)


def _production_candidate_frame(
    *,
    s1_ids: list[str],
    target_ids: list[str],
    scores: np.ndarray,
    indices: np.ndarray,
    source: int,
    country: str,
    view: str,
) -> pl.DataFrame:
    """Map S1 query rows and target-index positions to production pairs."""

    width = indices.shape[1]
    flat_indices = indices.reshape(-1)
    valid = flat_indices >= 0
    return pl.DataFrame(
        {
            "s1_id": np.repeat(np.asarray(s1_ids, dtype=object), width)[valid],
            "target_id": np.asarray(target_ids, dtype=object)[flat_indices[valid]],
            "source": f"S{source}",
            "country": country,
            "ann_view": view,
            "ann_rank": np.tile(np.arange(1, width + 1), len(s1_ids))[valid],
            "ann_score": scores.reshape(-1)[valid],
        }
    ).with_columns(
        pl.col("ann_rank").cast(pl.UInt16),
        pl.col("ann_score").cast(pl.Float32),
    )


def _remove_embedding_cache(
    cache_dir: Path, *, role: str, source: str, country: str, view: str
) -> None:
    array_path, metadata_path = _cache_paths(
        cache_dir,
        role=role,
        source=source,
        country=country,
        view=view,
    )
    array_path.unlink(missing_ok=True)
    metadata_path.unlink(missing_ok=True)
    array_path.with_suffix(".partial.npy").unlink(missing_ok=True)


def generate_ann_candidates(
    *,
    normalized_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    split: str,
    state: PipelineState,
    shard_count: int,
    views: tuple[str, ...] = VIEWS,
    top_k: int = 20,
    encoder_batch_size: int = 512,
    search_batch_size: int = 4096,
    index_type: str = "ivf",
    nlist: int = 4096,
    nprobe: int = 64,
    encoder_device: str = "cuda",
    faiss_device: str = "auto",
    keep_embeddings: bool = False,
    seed: int = 2026,
) -> dict[str, object]:
    if shard_count < 1 or top_k < 1:
        raise ValueError("shard_count and top_k must be positive")
    invalid_views = set(views) - set(VIEWS)
    if invalid_views:
        raise ValueError(f"unknown ANN views: {sorted(invalid_views)}")
    stage = f"ann_candidates_{split}"
    stage_signature = signature(
        {
            "stage": stage,
            "model": MODEL_NAME,
            "revision": MODEL_REVISION,
            "views": views,
            "top_k": top_k,
            "shard_count": shard_count,
            "index_type": index_type,
            "nlist": nlist,
            "nprobe": nprobe,
            "seed": seed,
            "version": 1,
        }
    )
    manifest = output_dir / "ann" / split / "manifest.json"
    if state.stage_complete(stage, stage_signature, (manifest,)):
        print(f"[{stage}] checkpoint complete; skipping", flush=True)
        return json.loads(manifest.read_text(encoding="utf-8"))

    state.begin_stage(stage, stage_signature)
    model = _load_encoder(encoder_device)
    s1_all = pl.read_parquet(source_output_path(normalized_dir, split, 1)).select(
        pl.col("entity_id").alias("s1_id"),
        "country",
        "business_name",
        "business_address",
    )
    countries = sorted(s1_all["country"].unique().to_list())
    counts: dict[str, int] = {}

    for country in countries:
        s1_country = s1_all.filter(pl.col("country") == country).rename(
            {"s1_id": "entity_id"}
        )
        for view in views:
            query_frame = _eligible_queries(s1_country, view)
            query_ids = query_frame["entity_id"].to_list()
            print(
                f"[{stage}] encoding S1/{country}/{view}: {query_frame.height:,}",
                flush=True,
            )
            query_embeddings = _encode_cached(
                query_frame,
                model=model,
                cache_dir=cache_dir,
                role="query",
                source="S1",
                country=country,
                view=view,
                batch_size=encoder_batch_size,
            )
            for source in (2, 3):
                target_all = pl.read_parquet(
                    source_output_path(normalized_dir, split, source)
                ).select("entity_id", "country", "business_name", "business_address")
                target_frame = _eligible_queries(
                    target_all.filter(pl.col("country") == country), view
                )
                if target_frame.is_empty() or query_frame.is_empty():
                    print(
                        f"[{stage}] no rows for S{source}/{country}/{view}",
                        flush=True,
                    )
                    continue
                target_ids = target_frame["entity_id"].to_list()
                print(
                    f"[{stage}] indexing S{source}/{country}/{view}: "
                    f"{target_frame.height:,} targets",
                    flush=True,
                )
                target_embeddings = _encode_cached(
                    target_frame,
                    model=model,
                    cache_dir=cache_dir,
                    role="database",
                    source=f"S{source}",
                    country=country,
                    view=view,
                    batch_size=encoder_batch_size,
                )
                index = _build_index(
                    target_embeddings,
                    index_type=index_type,
                    nlist=nlist,
                    nprobe=nprobe,
                    faiss_device=faiss_device,
                    seed=seed,
                )
                effective_top_k = min(top_k, target_frame.height)
                for shard in range(shard_count):
                    output = _ann_output_path(
                        output_dir, split, source, view, country, shard
                    )
                    shard_name = (
                        f"S{source}_{_safe_label(country)}_{view}_{shard:04d}"
                    )
                    shard_signature = signature(
                        {
                            "stage": stage_signature,
                            "source": source,
                            "country": country,
                            "view": view,
                            "shard": shard,
                        }
                    )
                    if state.shard_complete(
                        stage, shard_name, shard_signature, (output,)
                    ):
                        counts[shard_name] = int(
                            pl.scan_parquet(output).select(pl.len()).collect().item()
                        )
                        continue
                    positions = np.flatnonzero(
                        _shard_mask(query_ids, shard, shard_count)
                    )
                    shard_ids = [query_ids[position] for position in positions]
                    if len(positions):
                        scores, indices = _search(
                            index,
                            np.asarray(query_embeddings[positions]),
                            effective_top_k,
                            search_batch_size,
                        )
                        candidates = _production_candidate_frame(
                            s1_ids=shard_ids,
                            target_ids=target_ids,
                            scores=scores,
                            indices=indices,
                            source=source,
                            country=country,
                            view=view,
                        )
                        del scores, indices
                    else:
                        candidates = pl.DataFrame(
                            schema={
                                "s1_id": pl.String,
                                "target_id": pl.String,
                                "source": pl.String,
                                "country": pl.String,
                                "ann_view": pl.String,
                                "ann_rank": pl.UInt16,
                                "ann_score": pl.Float32,
                            }
                        )
                    _atomic_write_parquet(candidates, output)
                    counts[shard_name] = candidates.height
                    state.complete_shard(
                        stage,
                        shard_name,
                        shard_signature,
                        (output,),
                        {"rows": candidates.height},
                    )
                del index, target_embeddings, target_all, target_frame
                gc.collect()
                if not keep_embeddings:
                    _remove_embedding_cache(
                        cache_dir,
                        role="database",
                        source=f"S{source}",
                        country=country,
                        view=view,
                    )
            del query_embeddings
            gc.collect()
            if not keep_embeddings:
                _remove_embedding_cache(
                    cache_dir,
                    role="query",
                    source="S1",
                    country=country,
                    view=view,
                )

    result = {
        "split": split,
        "model": MODEL_NAME,
        "revision": MODEL_REVISION,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "views": list(views),
        "top_k_per_view": top_k,
        "shard_count": shard_count,
        "index_type": index_type,
        "nlist": nlist,
        "nprobe": nprobe,
        "rows_by_shard": counts,
        "total_rows": sum(counts.values()),
    }
    atomic_write_json(manifest, result)
    state.complete_stage(stage, (manifest,))
    return result
