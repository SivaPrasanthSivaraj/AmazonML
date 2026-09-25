"""Challenge-aligned parsing and evaluation metrics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping


def parse_id_list(value: str | None) -> frozenset[str]:
    """Parse the comma-separated ID-list format used by challenge TSV files.

    Empty strings and null values represent an empty set. Whitespace surrounding
    IDs is ignored, although generated submissions should not add such whitespace.
    """

    if value is None or not value.strip():
        return frozenset()
    return frozenset(item.strip() for item in value.split(",") if item.strip())


def entity_fbeta(
    truth: Iterable[str], prediction: Iterable[str], beta: float = 0.5
) -> float:
    """Return one S1 entity's set-based F-beta score.

    The explicit empty/empty case implements the challenge's singleton rule.
    """

    if beta <= 0:
        raise ValueError("beta must be positive")
    true_set = set(truth)
    predicted_set = set(prediction)
    if not true_set:
        return 1.0 if not predicted_set else 0.0
    if not predicted_set:
        return 0.0

    true_positives = len(true_set & predicted_set)
    if true_positives == 0:
        return 0.0
    precision = true_positives / len(predicted_set)
    recall = true_positives / len(true_set)
    beta_squared = beta * beta
    return (1.0 + beta_squared) * precision * recall / (
        beta_squared * precision + recall
    )


def macro_fbeta(
    truth: Mapping[str, Iterable[str]],
    prediction: Mapping[str, Iterable[str]],
    beta: float = 0.5,
) -> float:
    """Compute the official per-S1 macro-averaged F-beta score.

    Predictions must cover exactly the same S1 IDs as truth. Requiring exact
    coverage prevents an accidentally incomplete validation file from receiving a
    misleading score.
    """

    true_ids = set(truth)
    predicted_ids = set(prediction)
    if true_ids != predicted_ids:
        missing = len(true_ids - predicted_ids)
        unexpected = len(predicted_ids - true_ids)
        raise ValueError(
            "prediction S1 coverage differs from truth "
            f"(missing={missing}, unexpected={unexpected})"
        )
    if not true_ids:
        raise ValueError("cannot score an empty evaluation set")
    return sum(
        entity_fbeta(truth[s1_id], prediction[s1_id], beta=beta)
        for s1_id in true_ids
    ) / len(true_ids)
