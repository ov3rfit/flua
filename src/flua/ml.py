"""Utilities for transforming flua DataFrames into ML-ready feature matrices.

Typical workflow
----------------
1. Load sequences and build a DataFrame::

       from flua import load_multiple_fasta, groups_to_dataframe
       groups = load_multiple_fasta(["a.fasta", "b.fasta"])
       df, _ = groups_to_dataframe(groups, value_type="translated")

2. Check length consistency before encoding::

       from flua.ml import check_length_consistency
       print(check_length_consistency(df))

3. Choose an encoding strategy::

       from flua.ml import sequences_to_kmer_freq, encode_subtype
       from flua.encoding import sequences_to_ohe_tensor, PositionalOneHotEncoder

       X = sequences_to_kmer_freq(df, col="HA_aa", k=3)
       y, label_map = encode_subtype(df)

Available encodings
-------------------
Length-independent (no alignment required):

- :func:`sequences_to_composition` — character-frequency vector
- :func:`sequences_to_kmer_freq`   — k-mer frequency vector
- :func:`encode_subtype`           — label-encode a categorical target column

Fixed-length positional encodings (see :mod:`flua.encoding`):

- :func:`~flua.encoding.sequences_to_label_encoding` — integer per position
- :func:`~flua.encoding.sequences_to_ohe_tensor`     — 3-D one-hot tensor
- :class:`~flua.encoding.PositionalOneHotEncoder`    — flat 2-D sklearn matrix
"""

from __future__ import annotations

from itertools import product as _iterproduct
from typing import Literal

import pandas as pd

from flua.constants import AA_ALPHABET, NT_ALPHABET


def _infer_alphabet(series: pd.Series) -> str:
    """Guess nucleotide vs. amino-acid alphabet from a sequence column."""
    sample = series.dropna().head(20).str.upper().str.cat(sep="")
    chars = set(sample) - {"-", ".", "N", "X", "*"}
    if chars <= set(NT_ALPHABET + "U"):
        return NT_ALPHABET
    return AA_ALPHABET


def _seq_col_names(df: pd.DataFrame) -> list[str]:
    """Return sequence column names inferred from the ``_nt`` / ``_aa`` suffix
    convention used by :func:`~flua.io.groups_to_dataframe`."""
    return [c for c in df.columns if c.endswith("_nt") or c.endswith("_aa")]


# ---------------------------------------------------------------------------
# Length consistency check
# ---------------------------------------------------------------------------


def check_length_consistency(
    df: pd.DataFrame,
    seq_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Return per-column sequence-length statistics.

    Use this before any fixed-length encoding to decide on a safe ``length``
    value and to spot problematic samples.

    Parameters
    ----------
    df:
        DataFrame produced by :func:`~flua.io.groups_to_dataframe`, or any
        DataFrame whose columns contain string sequences.
    seq_cols:
        Columns to analyse.  When None, columns whose names end with
        ``_nt`` or ``_aa`` are used; if none are found, all ``object``-dtype
        columns are used.

    Returns
    -------
    pandas.DataFrame
        One row per column with the following fields:

        * ``seq_col``       — column name
        * ``count``         — number of non-null sequences
        * ``min_len``       — shortest sequence length
        * ``max_len``       — longest sequence length
        * ``mean_len``      — mean length (rounded to 1 decimal place)
        * ``std_len``       — standard deviation of lengths (0.0 when count <= 1)
        * ``is_consistent`` — ``True`` when all sequences share the same length
    """
    if seq_cols is None:
        seq_cols = _seq_col_names(df)
        if not seq_cols:
            seq_cols = df.select_dtypes(include="object").columns.tolist()

    rows: list[dict] = []
    for col in seq_cols:
        if col not in df.columns:
            continue
        lengths = df[col].dropna().str.len()
        if len(lengths) == 0:
            rows.append(
                {
                    "seq_col": col,
                    "count": 0,
                    "min_len": None,
                    "max_len": None,
                    "mean_len": None,
                    "std_len": None,
                    "is_consistent": True,
                }
            )
        else:
            rows.append(
                {
                    "seq_col": col,
                    "count": int(len(lengths)),
                    "min_len": int(lengths.min()),
                    "max_len": int(lengths.max()),
                    "mean_len": round(float(lengths.mean()), 1),
                    "std_len": round(float(lengths.std()), 2)
                    if len(lengths) > 1
                    else 0.0,
                    "is_consistent": bool(lengths.nunique() == 1),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Composition (character-frequency) encoding
# ---------------------------------------------------------------------------


def sequences_to_composition(
    df: pd.DataFrame,
    col: str,
    alphabet: str | None = None,
    normalize: bool = True,
) -> pd.DataFrame:
    """Convert a sequence column into a character-frequency (composition) matrix.

    Each character in the alphabet becomes one feature column.  The result is
    length-independent and works even when sequences vary in length across
    samples.

    Parameters
    ----------
    df:
        Source DataFrame.
    col:
        Name of the sequence column (e.g. ``"HA_aa"``).
    alphabet:
        Characters to count.  Auto-detected (nucleotide vs. amino acid) when None.
    normalize:
        When ``True`` (default), divide counts by sequence length to produce
        relative frequencies in [0, 1].  When ``False``, return raw counts.

    Returns
    -------
    pandas.DataFrame
        Shape ``(len(df), len(alphabet))``.  Column names are
        ``"{col}_{char}"``.  Rows with ``NaN`` sequences are all-zero.

    Examples
    --------
    >>> comp = sequences_to_composition(df, "HA_aa")
    >>> comp.shape
    (100, 20)
    """
    series = df[col]
    if alphabet is None:
        alphabet = _infer_alphabet(series)

    data: dict[str, list[float]] = {}
    for char in alphabet:

        def _count(s: object, c: str = char) -> float:
            if not isinstance(s, str) or len(s) == 0:
                return 0.0
            cnt = s.upper().count(c)
            return cnt / len(s) if normalize else float(cnt)

        data[f"{col}_{char}"] = [_count(s) for s in series]

    return pd.DataFrame(data, index=df.index)


# ---------------------------------------------------------------------------
# k-mer frequency encoding
# ---------------------------------------------------------------------------


def sequences_to_kmer_freq(
    df: pd.DataFrame,
    col: str,
    k: int = 3,
    alphabet: str | None = None,
    normalize: bool = True,
) -> pd.DataFrame:
    """Convert a sequence column into a k-mer frequency matrix.

    All ``len(alphabet)**k`` possible k-mers become feature columns.  Like
    :func:`sequences_to_composition`, the result is length-independent.
    k-mer frequencies capture local sequence patterns (e.g. amino-acid triplets
    for structure/function, trinucleotides for codon usage).

    Parameters
    ----------
    df:
        Source DataFrame.
    col:
        Name of the sequence column (e.g. ``"HA_aa"``).
    k:
        k-mer length.  Common choices:

        * ``k=1`` — equivalent to composition
        * ``k=2`` — dipeptide / dinucleotide frequencies
        * ``k=3`` — tripeptide / trinucleotide (codon) frequencies
    alphabet:
        Characters to enumerate.  Auto-detected when None.
    normalize:
        When ``True`` (default), divide each count by the total number of
        k-mers in the sequence.

    Returns
    -------
    pandas.DataFrame
        Shape ``(len(df), len(alphabet)**k)``.  Column names are
        ``"{col}_kmer_{kmer}"``.  Rows with ``NaN`` sequences are all-zero.

    Examples
    --------
    >>> kmer = sequences_to_kmer_freq(df, "HA_aa", k=3)
    >>> kmer.shape
    (100, 8000)   # 20**3 for amino acids
    """
    series = df[col]
    if alphabet is None:
        alphabet = _infer_alphabet(series)

    kmers: list[str] = ["".join(p) for p in _iterproduct(alphabet, repeat=k)]
    kmer_set = set(kmers)

    def _count_kmers(seq: object) -> dict[str, float]:
        zeros: dict[str, float] = dict.fromkeys(kmers, 0.0)
        if not isinstance(seq, str):
            return zeros
        seq = seq.upper()
        total = len(seq) - k + 1
        if total <= 0:
            return zeros
        counts: dict[str, float] = dict.fromkeys(kmers, 0.0)
        for i in range(total):
            km = seq[i : i + k]
            if km in kmer_set:
                counts[km] += 1.0
        if normalize:
            counts = {km: v / total for km, v in counts.items()}
        return counts

    rows = [_count_kmers(s) for s in series]
    result = pd.DataFrame(rows, index=df.index)
    result.columns = [f"{col}_kmer_{km}" for km in kmers]
    return result


# ---------------------------------------------------------------------------
# Subtype (target) encoding
# ---------------------------------------------------------------------------


def encode_subtype(
    df: pd.DataFrame,
    col: str = "subtype",
    strategy: Literal["label", "one_hot"] = "label",
) -> tuple[pd.Series | pd.DataFrame, dict[str, int]]:
    """Encode the subtype column for use as an ML target or categorical feature.

    Parameters
    ----------
    df:
        Source DataFrame.
    col:
        Column to encode (default ``"subtype"``).
    strategy:
        * ``"label"``   — map each subtype to a unique integer (classification
          with ordinal or embedding-based models).
        * ``"one_hot"`` — return a binary indicator DataFrame (good for
          multi-class softmax models or as input features).

    Returns
    -------
    tuple[pd.Series | pd.DataFrame, dict[str, int]]
        * First element: encoded target.  ``pd.Series`` for ``"label"``
          (``NaN`` -> ``-1``), ``pd.DataFrame`` for ``"one_hot"``
          (``NaN`` rows are all-zero).
        * Second element: ``{subtype_string: integer_label}`` mapping.

    Examples
    --------
    >>> y, label_map = encode_subtype(df)
    >>> label_map
    {'H1N1': 0, 'H3N2': 1, 'H5N1': 2}

    >>> y_ohe, label_map = encode_subtype(df, strategy="one_hot")
    >>> y_ohe.columns.tolist()
    ['subtype_H1N1', 'subtype_H3N2', 'subtype_H5N1']
    """
    unique_subtypes = sorted(df[col].dropna().unique())
    label_map: dict[str, int] = {st: i for i, st in enumerate(unique_subtypes)}

    if strategy == "label":
        encoded = df[col].map(label_map).fillna(-1).astype(int)
        return encoded, label_map

    # one_hot
    ohe_rows = []
    for val in df[col]:
        row = {f"{col}_{st}": 0 for st in unique_subtypes}
        if pd.notna(val) and val in label_map:
            row[f"{col}_{val}"] = 1
        ohe_rows.append(row)

    return pd.DataFrame(ohe_rows, index=df.index), label_map
