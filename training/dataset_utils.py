"""
Shared helpers for reading the CUAD ``master_clauses.csv`` file.

The structure of the CSV is detected dynamically:

* an *answer column* is any column whose name ends with "Answer", allowing
  variations such as ``"Renewal Term-Answer"`` and
  ``"Notice Period To Terminate Renewal- Answer"``;
* its *context column* is the column with the same (normalised) base name;
* a context column must predominantly contain Python-list literals such as
  ``[]`` or ``['clause text', ...]``;
* identifier columns (e.g. ``Filename``) are unpaired, non-list columns.

``Yes`` / ``No`` answers are annotation values, never clause text.
"""
from __future__ import annotations

import ast
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

ANSWER_COLUMN_PATTERN = re.compile(r"^(?P<base>.+?)\s*-\s*answer\s*$", re.IGNORECASE)
BINARY_ANSWERS = {"yes", "no"}


class DatasetError(Exception):
    """Raised when the dataset is missing, unreadable or structurally invalid."""


def normalize_name(name: str) -> str:
    """Case/whitespace-insensitive column-name key (``'Ip  Ownership'`` -> ``'ip ownership'``)."""
    name = re.sub(r"\s*-\s*", "-", str(name).strip().lower())
    return re.sub(r"\s+", " ", name)


def is_list_literal(value) -> bool:
    if not isinstance(value, str):
        return False
    v = value.strip()
    if not (v.startswith("[") and v.endswith("]")):
        return False
    try:
        return isinstance(ast.literal_eval(v), list)
    except (ValueError, SyntaxError):
        return False


def parse_context(value) -> List[str]:
    """Parse a CUAD context cell into a list of raw span strings."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    text = str(value).strip()
    if not text or text == "[]":
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        # Not a list literal: treat the raw string as a single span.
        return [text]
    if isinstance(parsed, (list, tuple)):
        return [str(p) for p in parsed if p is not None]
    return [str(parsed)]


def clean_answer(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def clean_clause_text(text: str) -> str:
    """Conservative legal-text cleaning.

    * normalises unicode, newlines and tabs;
    * replaces CUAD ``<omitted>`` markers with ``[...]``;
    * collapses excessive whitespace.

    Stop-words, negations (not, no, without, except, unless) and modal verbs
    (shall, must, may) are deliberately preserved.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    t = t.replace(config.CUAD_OMITTED_MARKER, config.OMITTED_REPLACEMENT)
    t = t.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    t = re.sub(r"\n+", " ", t)
    t = re.sub(r"[ \u00a0]{2,}", " ", t)
    t = re.sub(r"\s+([,.;:])", r"\1", t)
    return t.strip()


@dataclass
class CategoryPair:
    category: str
    context_column: str
    answer_column: str
    is_metadata: bool
    answer_type: str = "unknown"  # "binary" (Yes/No) or "value" (Nevada, 30 days, ...)


@dataclass
class DatasetStructure:
    n_rows: int
    n_columns: int
    identifier_columns: List[str]
    pairs: List[CategoryPair]
    unpaired_columns: List[str] = field(default_factory=list)

    @property
    def clause_pairs(self) -> List[CategoryPair]:
        return [p for p in self.pairs if not p.is_metadata]

    @property
    def metadata_pairs(self) -> List[CategoryPair]:
        return [p for p in self.pairs if p.is_metadata]

    @property
    def document_name_column(self) -> Optional[str]:
        for p in self.pairs:
            if normalize_name(p.category) == "document name":
                return p.answer_column
        return None


def load_raw_dataset(path: Path = config.RAW_DATASET_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DatasetError(
            f"Dataset not found at '{path}'. Place master_clauses.csv inside the data/ folder."
        )
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pandas raises many different error types
        raise DatasetError(f"Could not read '{path}' as CSV: {exc}") from exc
    if df.empty or df.shape[1] < 2:
        raise DatasetError(f"'{path}' is empty or has too few columns.")
    return df


def detect_structure(df: pd.DataFrame, list_ratio_threshold: float = 0.9) -> DatasetStructure:
    columns = list(df.columns)
    by_key: Dict[str, str] = {normalize_name(c): c for c in columns}
    metadata_keys = {normalize_name(m) for m in config.METADATA_CATEGORIES}

    pairs: List[CategoryPair] = []
    used = set()
    for col in columns:
        m = ANSWER_COLUMN_PATTERN.match(str(col))
        if not m:
            continue
        base_key = normalize_name(m.group("base"))
        ctx_col = by_key.get(base_key)
        if ctx_col is None or ctx_col == col:
            continue
        sample = df[ctx_col].dropna().astype(str)
        list_ratio = sample.map(is_list_literal).mean() if len(sample) else 0.0
        if list_ratio < list_ratio_threshold:
            continue
        answers = df[col].map(clean_answer)
        non_empty = answers[answers != ""]
        binary = len(non_empty) > 0 and non_empty.str.lower().isin(BINARY_ANSWERS).mean() > 0.95
        pairs.append(
            CategoryPair(
                category=str(ctx_col).strip(),
                context_column=ctx_col,
                answer_column=col,
                is_metadata=base_key in metadata_keys,
                answer_type="binary" if binary else "value",
            )
        )
        used.update({col, ctx_col})

    unpaired = [c for c in columns if c not in used]
    identifier_cols = []
    for c in unpaired:
        s = df[c].dropna().astype(str)
        if len(s) and s.map(is_list_literal).mean() < 0.5 and s.nunique() >= 0.95 * len(s):
            identifier_cols.append(c)
    if not pairs:
        raise DatasetError(
            "No clause/answer column pairs were detected. Expected columns such as "
            "'Governing Law' and 'Governing Law-Answer'."
        )
    return DatasetStructure(
        n_rows=len(df),
        n_columns=len(columns),
        identifier_columns=identifier_cols,
        pairs=pairs,
        unpaired_columns=[c for c in unpaired if c not in identifier_cols],
    )
