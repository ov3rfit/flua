"""Tests for flua.ml: ML-oriented feature encoding utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flua.ml import (
    AA_ALPHABET,
    NT_ALPHABET,
    check_length_consistency,
    encode_subtype,
    sequences_to_composition,
    sequences_to_kmer_freq,
    sequences_to_label_encoding,
    sequences_to_one_hot,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

AA_SEQS = ["MKTLL", "MKTLL", "MKTLL"]  # identical → consistent
NT_SEQS = ["ATGATG", "ATGATG", "ATGATG"]

MIXED_LENGTH_AA = ["MKT", "MKTLL", "MKTLLWW"]  # inconsistent lengths


def _df(seqs: list[str | None], col: str = "HA_aa") -> pd.DataFrame:
    return pd.DataFrame({col: seqs})


def _df_multi(
    subtype: list[str | None] | None = None,
) -> pd.DataFrame:
    data = {
        "HA_aa": AA_SEQS,
        "NA_aa": ["NKLL", "NKLL", "NKLL"],
        "subtype": subtype or ["H1N1", "H3N2", "H5N1"],
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# check_length_consistency
# ---------------------------------------------------------------------------


class TestCheckLengthConsistency:
    def test_returns_dataframe(self) -> None:
        df = _df(AA_SEQS)
        result = check_length_consistency(df)
        assert isinstance(result, pd.DataFrame)

    def test_consistent_sequences(self) -> None:
        df = _df(AA_SEQS)
        result = check_length_consistency(df)
        row = result.iloc[0]
        assert row["is_consistent"] == True  # noqa: E712
        assert row["min_len"] == row["max_len"] == 5
        assert row["std_len"] == 0.0

    def test_inconsistent_sequences(self) -> None:
        df = _df(MIXED_LENGTH_AA)
        result = check_length_consistency(df)
        row = result.iloc[0]
        assert row["is_consistent"] == False  # noqa: E712
        assert row["min_len"] == 3
        assert row["max_len"] == 7

    def test_nan_sequences_excluded_from_stats(self) -> None:
        df = _df(["MKTLL", None, "MKTLL"])
        result = check_length_consistency(df)
        assert result.iloc[0]["count"] == 2

    def test_all_nan_returns_zero_count(self) -> None:
        df = _df([None, None])
        result = check_length_consistency(df)
        assert result.iloc[0]["count"] == 0
        assert result.iloc[0]["is_consistent"] == True  # noqa: E712

    def test_auto_detects_seq_cols(self) -> None:
        df = _df_multi()
        result = check_length_consistency(df)
        assert set(result["seq_col"]) == {"HA_aa", "NA_aa"}
        # subtype is not a sequence column
        assert "subtype" not in result["seq_col"].values

    def test_explicit_seq_cols(self) -> None:
        df = _df_multi()
        result = check_length_consistency(df, seq_cols=["HA_aa"])
        assert list(result["seq_col"]) == ["HA_aa"]

    def test_unknown_col_skipped(self) -> None:
        df = _df(AA_SEQS)
        result = check_length_consistency(df, seq_cols=["nonexistent_col"])
        assert len(result) == 0

    def test_single_sequence_std_is_zero(self) -> None:
        df = _df(["MKTLL"])
        result = check_length_consistency(df)
        assert result.iloc[0]["std_len"] == 0.0

    def test_mean_len_correct(self) -> None:
        df = _df(["MKT", "MKTLL"])  # lengths 3, 5 → mean 4.0
        result = check_length_consistency(df)
        assert result.iloc[0]["mean_len"] == 4.0


# ---------------------------------------------------------------------------
# sequences_to_composition
# ---------------------------------------------------------------------------


class TestSequencesToComposition:
    def test_shape(self) -> None:
        df = _df(AA_SEQS)
        result = sequences_to_composition(df, "HA_aa", alphabet=AA_ALPHABET)
        assert result.shape == (3, len(AA_ALPHABET))

    def test_column_names(self) -> None:
        df = _df(["MKTLL"])
        result = sequences_to_composition(df, "HA_aa", alphabet="ACGT")
        assert list(result.columns) == ["HA_aa_A", "HA_aa_C", "HA_aa_G", "HA_aa_T"]

    def test_normalized_sums_to_one(self) -> None:
        df = _df(["AACCGGTT"], col="HA_nt")
        result = sequences_to_composition(df, "HA_nt", alphabet=NT_ALPHABET, normalize=True)
        row_sum = result.iloc[0].sum()
        assert abs(row_sum - 1.0) < 1e-6

    def test_unnormalized_counts(self) -> None:
        df = _df(["AAAC"], col="HA_nt")
        result = sequences_to_composition(
            df, "HA_nt", alphabet=NT_ALPHABET, normalize=False
        )
        assert result["HA_nt_A"].iloc[0] == 3.0
        assert result["HA_nt_C"].iloc[0] == 1.0

    def test_nan_row_is_zeros(self) -> None:
        df = _df([None, "MKTLL"])
        result = sequences_to_composition(df, "HA_aa", alphabet=AA_ALPHABET)
        assert result.iloc[0].sum() == 0.0

    def test_preserves_index(self) -> None:
        df = _df(AA_SEQS).set_index(pd.Index(["a", "b", "c"]))
        result = sequences_to_composition(df, "HA_aa", alphabet=AA_ALPHABET)
        assert list(result.index) == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# sequences_to_kmer_freq
# ---------------------------------------------------------------------------


class TestSequencesToKmerFreq:
    def test_shape_k1(self) -> None:
        df = _df(AA_SEQS)
        result = sequences_to_kmer_freq(df, "HA_aa", k=1, alphabet=AA_ALPHABET)
        assert result.shape == (3, len(AA_ALPHABET))

    def test_shape_k3_nucleotide(self) -> None:
        df = _df(NT_SEQS, col="HA_nt")
        result = sequences_to_kmer_freq(df, "HA_nt", k=3, alphabet=NT_ALPHABET)
        assert result.shape == (3, 4**3)

    def test_column_prefix(self) -> None:
        df = _df(AA_SEQS)
        result = sequences_to_kmer_freq(df, "HA_aa", k=1, alphabet="MK")
        assert all(c.startswith("HA_aa_kmer_") for c in result.columns)

    def test_normalized_rows_sum_to_one(self) -> None:
        df = _df(["ATGATG"], col="HA_nt")
        result = sequences_to_kmer_freq(
            df, "HA_nt", k=3, alphabet=NT_ALPHABET, normalize=True
        )
        row_sum = result.iloc[0].sum()
        assert abs(row_sum - 1.0) < 1e-6

    def test_unnormalized_counts(self) -> None:
        df = _df(["ATGATG"], col="HA_nt")
        result = sequences_to_kmer_freq(
            df, "HA_nt", k=3, alphabet=NT_ALPHABET, normalize=False
        )
        # ATGATG has k-mers: ATG, TGA, GAT, ATG  → ATG count = 2
        assert result["HA_nt_kmer_ATG"].iloc[0] == 2.0

    def test_nan_row_is_zeros(self) -> None:
        df = _df([None, "MKTLL"])
        result = sequences_to_kmer_freq(df, "HA_aa", k=2, alphabet=AA_ALPHABET)
        assert result.iloc[0].sum() == 0.0

    def test_sequence_shorter_than_k_is_zeros(self) -> None:
        df = _df(["AT"], col="HA_nt")
        result = sequences_to_kmer_freq(df, "HA_nt", k=3, alphabet=NT_ALPHABET)
        assert result.iloc[0].sum() == 0.0


# ---------------------------------------------------------------------------
# sequences_to_label_encoding
# ---------------------------------------------------------------------------


class TestSequencesToLabelEncoding:
    def test_shape(self) -> None:
        df = _df(AA_SEQS)
        result = sequences_to_label_encoding(df, "HA_aa", length=10, alphabet=AA_ALPHABET)
        assert result.shape == (3, 10)

    def test_dtype(self) -> None:
        df = _df(AA_SEQS)
        result = sequences_to_label_encoding(df, "HA_aa", length=5, alphabet=AA_ALPHABET)
        assert result.dtype == np.int16

    def test_known_chars_nonzero(self) -> None:
        df = _df(["M"])
        result = sequences_to_label_encoding(df, "HA_aa", length=5, alphabet=AA_ALPHABET)
        # 'M' is at index 9 in AA_ALPHABET (0-based) → encoded as 10 (1-based)
        assert result[0, 0] == AA_ALPHABET.index("M") + 1

    def test_padding_is_zero(self) -> None:
        df = _df(["M"])  # length 1, padded to 5
        result = sequences_to_label_encoding(df, "HA_aa", length=5, alphabet=AA_ALPHABET)
        assert result[0, 1] == 0
        assert result[0, 4] == 0

    def test_truncation(self) -> None:
        df = _df(["MKTLL"])  # length 5, truncated to 3
        result = sequences_to_label_encoding(df, "HA_aa", length=3, alphabet=AA_ALPHABET)
        assert result.shape[1] == 3

    def test_unknown_char_is_zero(self) -> None:
        df = _df(["MX"])  # 'X' not in AA_ALPHABET
        result = sequences_to_label_encoding(df, "HA_aa", length=5, alphabet=AA_ALPHABET)
        assert result[0, 1] == 0

    def test_nan_row_is_zeros(self) -> None:
        df = _df([None])
        result = sequences_to_label_encoding(df, "HA_aa", length=5, alphabet=AA_ALPHABET)
        assert (result[0] == 0).all()


# ---------------------------------------------------------------------------
# sequences_to_one_hot
# ---------------------------------------------------------------------------


class TestSequencesToOneHot:
    def test_shape(self) -> None:
        df = _df(AA_SEQS)
        result = sequences_to_one_hot(df, "HA_aa", length=5, alphabet=AA_ALPHABET)
        assert result.shape == (3, 5, len(AA_ALPHABET))

    def test_dtype(self) -> None:
        df = _df(AA_SEQS)
        result = sequences_to_one_hot(df, "HA_aa", length=5, alphabet=AA_ALPHABET)
        assert result.dtype == np.float32

    def test_known_position_is_one_hot(self) -> None:
        df = _df(["M"])
        result = sequences_to_one_hot(df, "HA_aa", length=5, alphabet=AA_ALPHABET)
        idx = AA_ALPHABET.index("M")
        assert result[0, 0, idx] == 1.0
        assert result[0, 0].sum() == 1.0

    def test_padding_positions_all_zero(self) -> None:
        df = _df(["M"])  # length 1, padded to 5
        result = sequences_to_one_hot(df, "HA_aa", length=5, alphabet=AA_ALPHABET)
        assert result[0, 1].sum() == 0.0
        assert result[0, 4].sum() == 0.0

    def test_truncation(self) -> None:
        df = _df(["MKTLL"])
        result = sequences_to_one_hot(df, "HA_aa", length=3, alphabet=AA_ALPHABET)
        assert result.shape == (1, 3, len(AA_ALPHABET))

    def test_unknown_char_all_zero(self) -> None:
        df = _df(["MX"])
        result = sequences_to_one_hot(df, "HA_aa", length=5, alphabet=AA_ALPHABET)
        assert result[0, 1].sum() == 0.0

    def test_nan_row_all_zero(self) -> None:
        df = _df([None])
        result = sequences_to_one_hot(df, "HA_aa", length=5, alphabet=AA_ALPHABET)
        assert result[0].sum() == 0.0

    def test_flatten_for_linear_model(self) -> None:
        df = _df(AA_SEQS)
        result = sequences_to_one_hot(df, "HA_aa", length=5, alphabet=AA_ALPHABET)
        flat = result.reshape(len(df), -1)
        assert flat.shape == (3, 5 * len(AA_ALPHABET))


# ---------------------------------------------------------------------------
# encode_subtype
# ---------------------------------------------------------------------------


class TestEncodeSubtype:
    def test_label_strategy_returns_series(self) -> None:
        df = _df_multi()
        encoded, label_map = encode_subtype(df, strategy="label")
        assert isinstance(encoded, pd.Series)

    def test_label_map_keys(self) -> None:
        df = _df_multi()
        _, label_map = encode_subtype(df)
        assert set(label_map.keys()) == {"H1N1", "H3N2", "H5N1"}

    def test_label_values_are_unique_ints(self) -> None:
        df = _df_multi()
        _, label_map = encode_subtype(df)
        values = list(label_map.values())
        assert len(values) == len(set(values))
        assert all(isinstance(v, int) for v in values)

    def test_nan_encoded_as_minus_one(self) -> None:
        df = _df_multi(subtype=["H1N1", None, "H5N1"])
        encoded, _ = encode_subtype(df, strategy="label")
        assert encoded.iloc[1] == -1

    def test_one_hot_strategy_returns_dataframe(self) -> None:
        df = _df_multi()
        encoded, _ = encode_subtype(df, strategy="one_hot")
        assert isinstance(encoded, pd.DataFrame)

    def test_one_hot_shape(self) -> None:
        df = _df_multi()
        encoded, _ = encode_subtype(df, strategy="one_hot")
        assert encoded.shape == (3, 3)  # 3 samples, 3 unique subtypes

    def test_one_hot_column_names(self) -> None:
        df = _df_multi()
        encoded, _ = encode_subtype(df, strategy="one_hot")
        assert "subtype_H1N1" in encoded.columns

    def test_one_hot_nan_row_all_zero(self) -> None:
        df = _df_multi(subtype=["H1N1", None, "H5N1"])
        encoded, _ = encode_subtype(df, strategy="one_hot")
        assert encoded.iloc[1].sum() == 0

    def test_one_hot_known_row_sums_to_one(self) -> None:
        df = _df_multi()
        encoded, _ = encode_subtype(df, strategy="one_hot")
        assert encoded.iloc[0].sum() == 1

    def test_custom_col(self) -> None:
        df = pd.DataFrame({"label": ["A", "B", "A"]})
        encoded, label_map = encode_subtype(df, col="label")
        assert set(label_map.keys()) == {"A", "B"}
        assert encoded.iloc[0] == encoded.iloc[2]


# ---------------------------------------------------------------------------
# Host encoding integration
# ---------------------------------------------------------------------------


class TestHostEncoding:
    """Verify that host information round-trips correctly through encode_subtype."""

    def _host_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "HA_aa": ["MKTLL", "MKTLL", "MKTLL", "MKTLL"],
                "subtype": ["H1N1", "H1N1", "H3N2", "H3N2"],
                "host": ["Human", "Swine", "Human", None],
            }
        )

    def test_host_label_encoding(self) -> None:
        df = self._host_df()
        encoded, label_map = encode_subtype(df, col="host", strategy="label")
        assert set(label_map.keys()) == {"Human", "Swine"}
        assert encoded.iloc[3] == -1  # None → -1

    def test_host_label_encoding_consistent(self) -> None:
        df = self._host_df()
        encoded, label_map = encode_subtype(df, col="host", strategy="label")
        # Both Human rows should get the same label
        assert encoded.iloc[0] == encoded.iloc[2] == label_map["Human"]

    def test_host_one_hot_shape(self) -> None:
        df = self._host_df()
        encoded, _ = encode_subtype(df, col="host", strategy="one_hot")
        assert encoded.shape == (4, 2)  # 4 samples, 2 unique hosts

    def test_host_one_hot_column_names(self) -> None:
        df = self._host_df()
        encoded, _ = encode_subtype(df, col="host", strategy="one_hot")
        assert "host_Human" in encoded.columns
        assert "host_Swine" in encoded.columns

    def test_host_one_hot_nan_row_all_zero(self) -> None:
        df = self._host_df()
        encoded, _ = encode_subtype(df, col="host", strategy="one_hot")
        assert encoded.iloc[3].sum() == 0  # None row is all-zero

    def test_host_one_hot_known_row_sums_to_one(self) -> None:
        df = self._host_df()
        encoded, _ = encode_subtype(df, col="host", strategy="one_hot")
        assert encoded.iloc[0].sum() == 1
        assert encoded.iloc[1].sum() == 1

    def test_combined_subtype_and_host_features(self) -> None:
        """Subtype and host one-hot features can be concatenated for ML."""
        df = self._host_df()
        subtype_ohe, _ = encode_subtype(df, col="subtype", strategy="one_hot")
        host_ohe, _ = encode_subtype(df, col="host", strategy="one_hot")
        combined = pd.concat([subtype_ohe, host_ohe], axis=1)
        assert combined.shape == (4, 4)  # 2 subtype + 2 host columns
