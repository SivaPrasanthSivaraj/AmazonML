"""Train, calibrate, and persist the leakage-safe production matcher."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from .metrics import entity_fbeta
from .pipeline_state import PipelineState, atomic_write_json, signature
from .production_data import source_output_path, truth_output_path
from .production_features import SAFE_BASE_FEATURES


def _feature_paths(features_dir: Path, split: str) -> list[Path]:
    paths = sorted((features_dir / "features" / split).glob("S*/part-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no feature shards below {features_dir / 'features' / split}")
    return paths


def _load_features(features_dir: Path, split: str) -> pl.DataFrame:
    return pl.scan_parquet(_feature_paths(features_dir, split)).collect(engine="streaming")


def _truth_maps(
    normalized_dir: Path,
) -> tuple[dict[str, set[str]], dict[str, str]]:
    s1 = pl.read_parquet(source_output_path(normalized_dir, "train", 1)).select(
        pl.col("entity_id").alias("s1_id"), pl.col("split").alias("validation_split")
    )
    truth = pl.read_parquet(truth_output_path(normalized_dir)).select(
        "s1_id", "target_id"
    )
    truth_rows = truth.group_by("s1_id").agg("target_id")
    truth_by_s1 = {s1_id: set(targets) for s1_id, targets in truth_rows.iter_rows()}
    split_by_s1 = dict(s1.iter_rows())
    for s1_id in split_by_s1:
        truth_by_s1.setdefault(s1_id, set())
    return truth_by_s1, split_by_s1


def _macro_result(
    frame: pl.DataFrame,
    scores: np.ndarray,
    threshold: float,
    validation_split: str,
    truth_by_s1: dict[str, set[str]],
    split_by_s1: dict[str, str],
) -> dict[str, object]:
    predicted = (
        frame.select("s1_id", "target_id")
        .with_columns(pl.Series("score", scores))
        .filter(pl.col("score") >= threshold)
        .group_by("s1_id")
        .agg(pl.col("target_id").unique())
    )
    predicted_by_s1 = {
        s1_id: set(targets) for s1_id, targets in predicted.iter_rows()
    }
    entity_ids = [
        s1_id for s1_id, value in split_by_s1.items() if value == validation_split
    ]
    all_scores: list[float] = []
    singleton_scores: list[float] = []
    linked_scores: list[float] = []
    for s1_id in entity_ids:
        truth = truth_by_s1[s1_id]
        score = entity_fbeta(truth, predicted_by_s1.get(s1_id, set()), beta=0.5)
        all_scores.append(score)
        (linked_scores if truth else singleton_scores).append(score)
    return {
        "threshold": float(threshold),
        "macro_f0_5": float(np.mean(all_scores)),
        "singleton_f0_5": float(np.mean(singleton_scores)),
        "linked_f0_5": float(np.mean(linked_scores)),
        "entities": len(entity_ids),
        "predicted_links": int(sum(len(value) for value in predicted_by_s1.values())),
    }


def _candidate_recall(
    frame: pl.DataFrame,
    validation_split: str,
    truth_by_s1: dict[str, set[str]],
    split_by_s1: dict[str, str],
) -> float:
    retrieved = {
        (s1_id, target_id)
        for s1_id, target_id in frame.select("s1_id", "target_id").iter_rows()
    }
    truth = {
        (s1_id, target_id)
        for s1_id, targets in truth_by_s1.items()
        if split_by_s1[s1_id] == validation_split
        for target_id in targets
    }
    return len(retrieved & truth) / len(truth) if truth else 0.0


def train_production_model(
    *,
    normalized_dir: Path,
    features_dir: Path,
    output_dir: Path,
    state: PipelineState,
    seed: int = 2026,
) -> dict[str, object]:
    try:
        import lightgbm as lgb
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ImportError as error:
        raise RuntimeError("install LightGBM and scikit-learn before model training") from error

    stage = "train_model"
    stage_signature = signature(
        {"stage": stage, "features": SAFE_BASE_FEATURES, "seed": seed, "version": 1}
    )
    model_dir = output_dir / "model"
    model_path = model_dir / "final_master_model.txt"
    metadata_path = model_dir / "master_metadata.json"
    importance_path = model_dir / "feature_importance.csv"
    outputs = (model_path, metadata_path, importance_path)
    if state.stage_complete(stage, stage_signature, outputs):
        print("[train_model] checkpoint complete; skipping", flush=True)
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    state.begin_stage(stage, stage_signature)

    data = _load_features(features_dir, "train")
    train = data.filter(pl.col("validation_split") == "train")
    calibration = data.filter(pl.col("validation_split") == "calibration")
    holdout = data.filter(pl.col("validation_split") == "holdout")
    print(
        f"[train_model] rows train={train.height:,} calibration={calibration.height:,} "
        f"holdout={holdout.height:,}",
        flush=True,
    )
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1600,
        learning_rate=0.025,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.90,
        reg_alpha=0.10,
        reg_lambda=2.0,
        scale_pos_weight=2.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        train.select(SAFE_BASE_FEATURES).to_numpy(),
        train["is_match"].cast(pl.UInt8).to_numpy(),
        eval_set=[
            (
                calibration.select(SAFE_BASE_FEATURES).to_numpy(),
                calibration["is_match"].cast(pl.UInt8).to_numpy(),
            )
        ],
        eval_metric=["auc", "binary_logloss"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)],
        feature_name=list(SAFE_BASE_FEATURES),
    )
    best_iteration = int(model.best_iteration_)
    calibration_scores = model.predict_proba(
        calibration.select(SAFE_BASE_FEATURES).to_numpy(),
        num_iteration=best_iteration,
    )[:, 1]
    holdout_scores = model.predict_proba(
        holdout.select(SAFE_BASE_FEATURES).to_numpy(),
        num_iteration=best_iteration,
    )[:, 1]
    truth_by_s1, split_by_s1 = _truth_maps(normalized_dir)
    thresholds = np.unique(
        np.concatenate([np.linspace(0.05, 0.90, 86), np.linspace(0.905, 0.995, 91)])
    )
    calibration_results = [
        _macro_result(
            calibration,
            calibration_scores,
            threshold,
            "calibration",
            truth_by_s1,
            split_by_s1,
        )
        for threshold in thresholds
    ]
    best_calibration = max(
        calibration_results, key=lambda value: value["macro_f0_5"]
    )
    holdout_result = _macro_result(
        holdout,
        holdout_scores,
        float(best_calibration["threshold"]),
        "holdout",
        truth_by_s1,
        split_by_s1,
    )
    ranking = {
        "calibration_average_precision": float(
            average_precision_score(calibration["is_match"], calibration_scores)
        ),
        "holdout_average_precision": float(
            average_precision_score(holdout["is_match"], holdout_scores)
        ),
        "holdout_roc_auc": float(
            roc_auc_score(holdout["is_match"], holdout_scores)
        ),
    }
    candidate_recall = {
        "calibration": _candidate_recall(
            calibration, "calibration", truth_by_s1, split_by_s1
        ),
        "holdout": _candidate_recall(
            holdout, "holdout", truth_by_s1, split_by_s1
        ),
    }

    print(f"[train_model] calibration {best_calibration}", flush=True)
    print(f"[train_model] holdout {holdout_result}", flush=True)
    final_model = lgb.LGBMClassifier(**model.get_params())
    final_model.set_params(n_estimators=best_iteration)
    final_model.fit(
        data.select(SAFE_BASE_FEATURES).to_numpy(),
        data["is_match"].cast(pl.UInt8).to_numpy(),
        feature_name=list(SAFE_BASE_FEATURES),
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    final_model.booster_.save_model(model_path)
    importance = pl.DataFrame(
        {
            "feature": SAFE_BASE_FEATURES,
            "gain": model.booster_.feature_importance(importance_type="gain"),
        }
    ).sort("gain", descending=True)
    importance.write_csv(importance_path)
    metadata = {
        "features": list(SAFE_BASE_FEATURES),
        "best_iteration": best_iteration,
        "threshold": float(best_calibration["threshold"]),
        "ranking": ranking,
        "candidate_recall": candidate_recall,
        "calibration": best_calibration,
        "holdout": holdout_result,
        "leakage_policy": "Only direct lexical/context/E5 pair features are modeled.",
    }
    atomic_write_json(metadata_path, metadata)
    state.complete_stage(stage, outputs)
    return metadata
