"""Conservative, Unicode-aware text normalization.

Raw fields must always be retained. These normalized views are retrieval and model
features; none is treated as a definitive business identifier.
"""

from __future__ import annotations

import re
import unicodedata

import polars as pl


_SPACE_RE = re.compile(r"\s+")


def normalize_text(value: str | None, *, strip_marks: bool = False) -> str:
    """Normalize punctuation/case while preserving Unicode letters and numbers."""

    if not value:
        return ""
    form = "NFKD" if strip_marks else "NFKC"
    text = unicodedata.normalize(form, value).casefold()
    if strip_marks:
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = "".join(
        ch
        if ch.isalnum() or (not strip_marks and unicodedata.category(ch).startswith("M"))
        else " "
        for ch in text
    )
    return _SPACE_RE.sub(" ", text).strip()


def normalized_expr(column: str, *, strip_marks: bool = False) -> pl.Expr:
    """Polars equivalent used for streaming normalization of full TSV files."""

    expr = pl.col(column).fill_null("").str.normalize("NFKD" if strip_marks else "NFKC")
    if strip_marks:
        expr = expr.str.replace_all(r"\p{M}+", "")
    return (
        expr.str.to_lowercase()
        .str.replace_all(
            r"[^\p{L}\p{N}]+" if strip_marks else r"[^\p{L}\p{N}\p{M}]+",
            " ",
        )
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def compact_expr(column: str, *, strip_marks: bool = False) -> pl.Expr:
    """Return a separator-free normalized expression for exact retrieval."""

    return normalized_expr(column, strip_marks=strip_marks).str.replace_all(" ", "")
