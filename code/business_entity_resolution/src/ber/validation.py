"""Deterministic S1-level validation assignment."""

from __future__ import annotations

import polars as pl


SPLIT_NAMES = ("train", "calibration", "holdout")


def split_expr(entity_id_column: str = "entity_id") -> pl.Expr:
    """Assign random-looking numeric S1 IDs to a stable 80/10/10 split.

    Challenge IDs have a random numeric suffix. Modulo assignment is reproducible,
    cheap on millions of rows, and keeps every S1 entity (and all of its links) in
    exactly one partition.
    """

    bucket = (
        pl.col(entity_id_column)
        .str.strip_prefix("S1-")
        .cast(pl.UInt64)
        .mod(100)
    )
    return (
        pl.when(bucket < 80)
        .then(pl.lit("train"))
        .when(bucket < 90)
        .then(pl.lit("calibration"))
        .otherwise(pl.lit("holdout"))
    )
