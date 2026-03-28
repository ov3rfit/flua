"""Positional sequence encoders for fixed-length amino-acid columns.

This module groups all encoders that operate position-by-position on
fixed-length sequences.  Use :func:`~flua.ml.check_length_consistency`
first to verify that your sequences are uniformly aligned before calling
any function here.

Available encoders
------------------
- :func:`sequences_to_label_encoding` — integer per position (embedding layers)
- :func:`sequences_to_ohe_tensor`     — 3-D one-hot tensor (CNN / Transformer)
- :class:`PositionalOneHotEncoder`    — flat 2-D feature matrix (scikit-learn)

Choosing the right encoder
---------------------------
All three encoders represent the same information (which residue appears at
each position), but in different shapes suited to different model families:

sequences_to_label_encoding
    Returns ``(n_samples, length)`` int16.  Each position is a single integer
    (1-based index into the alphabet, 0 for unknown/padding).  Feed this into
    an embedding layer (``torch.nn.Embedding``, ``keras.layers.Embedding``).

sequences_to_ohe_tensor
    Returns ``(n_samples, length, alphabet_size)`` float32.  Each position is
    a one-hot binary vector.  Feed this directly into a Conv1d or Transformer
    encoder that expects a sequence of per-position feature vectors.

PositionalOneHotEncoder
    Returns a flat ``(n_samples, n_features)`` DataFrame where every
    (position, residue) pair is an independent named column (e.g. ``pb2_159K``).
    Encodes multiple segment columns in one step, and a ``shrink`` option drops
    features never observed in training to keep the matrix compact.  Use this
    with any scikit-learn estimator (RandomForest, SVM, logistic regression, …).
"""

from __future__ import annotations

from typing import Literal  # noqa: F401 — kept for potential future type hints

import numpy as np
import pandas as pd

from flua.constants import AA_ALPHABET_EXTENDED
from flua.ml import _infer_alphabet


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _seqs_to_char_matrix(series: pd.Series, seq_len: int) -> np.ndarray:
    """Convert a sequence Series to a 2-D character array of shape ``(n, seq_len)``.

    * Sequences are upper-cased and truncated to seq_len.
    * Sequences shorter than seq_len are right-padded with empty strings.
    * ``NaN`` / non-string rows become rows of empty strings.

    The returned dtype is ``<U1`` (single Unicode character per cell).
    Empty-string cells will not match any standard residue character, so they
    are effectively treated as unknown / missing by downstream encoders.

    Shared by :func:`sequences_to_label_encoding`, :func:`sequences_to_ohe_tensor`,
    and :class:`PositionalOneHotEncoder`.
    """
    n = len(series)
    mat = np.empty((n, seq_len), dtype="<U1")
    mat[:] = ""
    for i, val in enumerate(series):
        if isinstance(val, str) and len(val) > 0:
            chars = val.upper()[:seq_len]
            mat[i, : len(chars)] = list(chars)
    return mat


def _aa_col_names(df: pd.DataFrame) -> list[str]:
    """Return column names that end with ``_aa`` (amino-acid sequence columns)."""
    return [c for c in df.columns if c.endswith("_aa")]


def _segment_name(col: str) -> str:
    """Strip the ``_aa`` suffix to get the segment label (e.g. ``pb2_aa`` -> ``pb2``)."""
    return col[:-3] if col.endswith("_aa") else col


# ---------------------------------------------------------------------------
# Integer (label) encoding
# ---------------------------------------------------------------------------


def sequences_to_label_encoding(
    df: pd.DataFrame,
    col: str,
    length: int,
    alphabet: str | None = None,
) -> np.ndarray:
    """Convert a sequence column into an integer-encoded 2-D array.

    Each character maps to a positive integer (1-based index into the
    alphabet).  Sequences are right-padded with zeros or truncated to exactly
    length positions.  Unknown characters and padding positions are encoded
    as ``0``.

    Use this encoding when feeding sequences to an embedding layer in a
    deep-learning model (``torch.nn.Embedding``, ``keras.layers.Embedding``).

    .. tip::
       Use :func:`~flua.ml.check_length_consistency` first to choose a safe
       length value (e.g. the maximum observed length).

    .. note::
       This function encodes one column at a time and returns a raw integer
       array intended for embedding layers.  For a flat feature matrix suitable
       for scikit-learn models, see :class:`PositionalOneHotEncoder`.

    Parameters
    ----------
    df:
        Source DataFrame.
    col:
        Name of the sequence column.
    length:
        Output sequence length (truncate if longer, zero-pad if shorter).
    alphabet:
        Characters to map.  Auto-detected (nucleotide vs. amino acid) when None.

    Returns
    -------
    numpy.ndarray
        Shape ``(len(df), length)``, dtype ``int16``.

    Examples
    --------
    >>> X = sequences_to_label_encoding(df, "HA_aa", length=566)
    >>> X.shape
    (100, 566)
    """
    series = df[col]
    if alphabet is None:
        alphabet = _infer_alphabet(series)

    char_mat = _seqs_to_char_matrix(series, length)
    result = np.zeros((len(df), length), dtype=np.int16)
    for i, char in enumerate(alphabet):
        result[char_mat == char] = i + 1  # 1-based; 0 stays for unknown/padding

    return result


# ---------------------------------------------------------------------------
# One-hot tensor encoding
# ---------------------------------------------------------------------------


def sequences_to_ohe_tensor(
    df: pd.DataFrame,
    col: str,
    length: int,
    alphabet: str | None = None,
) -> np.ndarray:
    """Convert a sequence column into a 3-D one-hot tensor for deep-learning models.

    Each position in the sequence becomes a binary vector of length
    ``len(alphabet)``, producing a tensor of shape
    ``(n_samples, seq_length, alphabet_size)``.  This layout matches the
    ``(batch, length, channels)`` convention used by PyTorch ``Conv1d`` /
    ``TransformerEncoder`` and TensorFlow/Keras ``Conv1D`` / ``MultiHeadAttention``
    layers.

    Unknown characters and padding positions are represented as all-zero vectors.

    .. note::
       Choosing between the three positional encoders:

       * Use :func:`sequences_to_label_encoding` when your model starts with an
         embedding layer that maps each integer token to a dense vector.

       * Use :func:`sequences_to_ohe_tensor` when your CNN or Transformer expects
         a direct one-hot input without a learnable embedding.

       * Use :class:`PositionalOneHotEncoder` when your model is a scikit-learn
         estimator that requires a flat 2-D feature matrix with named columns.

    Parameters
    ----------
    df:
        Source DataFrame.
    col:
        Name of the sequence column.
    length:
        Fixed sequence length.  Sequences longer than length are truncated;
        shorter sequences and ``NaN`` rows are zero-padded.
    alphabet:
        Characters to encode.  Auto-detected (nucleotide vs. amino acid) when None.

    Returns
    -------
    numpy.ndarray
        Shape ``(len(df), length, len(alphabet))``, dtype ``float32``.

    Examples
    --------
    >>> X = sequences_to_ohe_tensor(df, "HA_aa", length=566)
    >>> X.shape
    (100, 566, 20)

    Feed directly into a PyTorch Conv1d (expects ``(batch, channels, length)``):

    >>> import torch
    >>> t = torch.from_numpy(X).permute(0, 2, 1)  # -> (100, 20, 566)

    Flatten to a 2-D matrix for a simple linear model (prefer
    :class:`PositionalOneHotEncoder` for this use-case — it adds
    named columns and a ``shrink`` option):

    >>> X_flat = X.reshape(len(df), -1)   # (100, 11320)
    """
    series = df[col]
    if alphabet is None:
        alphabet = _infer_alphabet(series)
    n_chars = len(alphabet)

    char_mat = _seqs_to_char_matrix(series, length)
    result = np.zeros((len(df), length, n_chars), dtype=np.float32)
    for i, char in enumerate(alphabet):
        result[:, :, i] = (char_mat == char).astype(np.float32)

    return result


# ---------------------------------------------------------------------------
# Positional one-hot encoder (scikit-learn compatible)
# ---------------------------------------------------------------------------


class PositionalOneHotEncoder:
    """Positional one-hot encoder for fixed-length amino-acid sequence columns.

    Encodes one or more ``_aa`` segment columns into a flat 2-D pandas
    DataFrame whose columns are named ``{segment}_{pos}{residue}``
    (e.g. ``pb2_159K``).  The output is directly usable as the feature matrix
    ``X`` for any scikit-learn estimator.

    .. rubric:: When to use this class vs the functional encoders

    Use :class:`PositionalOneHotEncoder` when your downstream model is a
    scikit-learn estimator (RandomForest, SVM, logistic regression, …)
    that requires a flat 2-D feature matrix.  The class handles multiple
    segment columns, produces human-readable column names, and supports
    ``shrink`` to keep the matrix compact.

    Use :func:`sequences_to_ohe_tensor` instead when your model is a CNN or
    Transformer that needs to treat each sequence position as a spatial step.
    Use :func:`sequences_to_label_encoding` when your model uses an embedding
    layer.  Both functional encoders return arrays and encode one column at a time.

    Parameters
    ----------
    aa_columns:
        Sequence column names to encode.  When None, all columns whose names
        end with ``_aa`` are detected automatically.
    alphabet:
        Characters considered as valid residues.  Defaults to the 20 standard
        amino acids plus ``*`` (stop codon) and ``X`` (ambiguous residue).
        Characters not in the alphabet are silently treated as unknown
        (encoded as an all-zero vector).
    shrink:
        When ``True`` (default), drop any (position, residue) feature that was
        never observed in the data passed to :meth:`fit`.  This substantially
        reduces the number of output columns without losing any information
        present in the training set.

    Attributes
    ----------
    feature_names_out_ : list[str]
        Feature names in column order, available after :meth:`fit`.
        Each name follows the ``{segment}_{pos}{residue}`` convention.
    columns_ : list[str]
        Sequence columns that were actually encoded.
    seq_lengths_ : dict[str, int]
        ``{column: length}`` mapping determined during :meth:`fit`.

    Examples
    --------
    Basic usage::

        enc = PositionalOneHotEncoder(shrink=True)
        X_train = enc.fit_transform(train_df)
        X_test  = enc.transform(test_df)

    Inspect the most discriminative positions after training a classifier::

        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier().fit(X_train, y_train)
        importances = pd.Series(clf.feature_importances_,
                                index=enc.feature_names_out_)
        print(importances.nlargest(10))
    """

    def __init__(
        self,
        aa_columns: list[str] | None = None,
        alphabet: str | None = None,
        shrink: bool = True,
    ) -> None:
        self.aa_columns = aa_columns
        self.alphabet = alphabet if alphabet is not None else AA_ALPHABET_EXTENDED
        self.shrink = shrink

        # Set after fit
        self.columns_: list[str] = []
        self.seq_lengths_: dict[str, int] = {}
        self.feature_names_out_: list[str] = []
        # Per-column list of (0-based pos, residue) pairs to include in output
        self._active_features_: dict[str, list[tuple[int, str]]] = {}
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "PositionalOneHotEncoder":
        """Fit the encoder on df: determine sequence lengths and active features.

        When ``shrink=True``, only (position, residue) combinations that
        actually appear in df will be included in the output.  Pass the
        training DataFrame here so that the feature set is fixed before
        the test data are seen.

        Parameters
        ----------
        df:
            DataFrame containing the ``_aa`` sequence columns.

        Returns
        -------
        self
        """
        cols = self.aa_columns if self.aa_columns is not None else _aa_col_names(df)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in df: {missing}")

        self.columns_ = cols
        self._active_features_ = {}
        self.feature_names_out_ = []
        self.seq_lengths_ = {}

        alphabet_set = set(self.alphabet)

        for col in cols:
            series = df[col].dropna()
            if len(series) == 0:
                self.seq_lengths_[col] = 0
                self._active_features_[col] = []
                continue

            seq_len = len(series.iloc[0])
            self.seq_lengths_[col] = seq_len

            if seq_len == 0:
                self._active_features_[col] = []
                continue

            char_mat = _seqs_to_char_matrix(series, seq_len)
            seg = _segment_name(col)
            active: list[tuple[int, str]] = []

            for pos in range(seq_len):
                if self.shrink:
                    observed = set(char_mat[:, pos].tolist()) & alphabet_set
                else:
                    observed = alphabet_set
                for res in sorted(observed):
                    active.append((pos, res))
                    self.feature_names_out_.append(f"{seg}_{pos + 1}{res}")

            self._active_features_[col] = active

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode sequences in df using the vocabulary fitted on training data.

        The output always has the same columns as determined by :meth:`fit`,
        regardless of which residues happen to appear in df.  This guarantees
        that training and test matrices share the same feature layout.

        Sequences longer than the fitted length are truncated; shorter sequences
        and ``NaN`` rows are zero-padded.

        Parameters
        ----------
        df:
            DataFrame with the same sequence columns used during :meth:`fit`.

        Returns
        -------
        pandas.DataFrame
            Shape ``(len(df), len(feature_names_out_))``, dtype ``uint8``.
            Index is preserved from df.
        """
        if not self._fitted:
            raise RuntimeError(
                "Call fit() before transform().  "
                "Use fit_transform() to fit and transform in one step."
            )

        n = len(df)
        blocks: list[np.ndarray] = []

        for col in self.columns_:
            active = self._active_features_[col]
            seq_len = self.seq_lengths_[col]

            if not active or seq_len == 0:
                continue

            char_mat = _seqs_to_char_matrix(df[col], seq_len)
            col_block = np.zeros((n, len(active)), dtype=np.uint8)
            for feat_i, (pos, res) in enumerate(active):
                col_block[:, feat_i] = (char_mat[:, pos] == res).view(np.uint8)

            blocks.append(col_block)

        if not blocks:
            return pd.DataFrame(index=df.index)

        matrix = np.concatenate(blocks, axis=1)
        return pd.DataFrame(matrix, index=df.index, columns=self.feature_names_out_)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on df and return the encoded matrix in one step.

        Equivalent to ``fit(df).transform(df)``.  Pass only the training
        DataFrame; transform the test DataFrame separately with
        :meth:`transform` to avoid data leakage.
        """
        return self.fit(df).transform(df)

    def get_feature_names_out(self) -> list[str]:
        """Return feature names (compatible with the scikit-learn ``set_output`` API)."""
        return self.feature_names_out_

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        n_feat = len(self.feature_names_out_) if self._fitted else "?"
        return (
            f"PositionalOneHotEncoder("
            f"shrink={self.shrink}, "
            f"alphabet_size={len(self.alphabet)}, "
            f"status={status}, "
            f"n_features={n_feat})"
        )
