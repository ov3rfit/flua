"""Tests for flua.encoding: positional sequence encoders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flua.constants import AA_ALPHABET, AA_ALPHABET_EXTENDED
from flua.encoding import (
    PositionalOneHotEncoder,
    _segment_name,
    sequences_to_label_encoding,
    sequences_to_ohe_tensor,
)

# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

PB2_LEN = 5
HA_LEN = 4

SEQ_PB2 = ["MKTLL", "MKTLV", "MKTLK"]
SEQ_HA = ["NACD", "NACD", "NACM"]


def _df(
    pb2: list[str | None] | None = None,
    ha: list[str | None] | None = None,
    host: list[str] | None = None,
    n: int = 3,
) -> pd.DataFrame:
    data: dict = {}
    if pb2 is not None:
        data["pb2_aa"] = pb2
    if ha is not None:
        data["ha_aa"] = ha
    if host is not None:
        data["host"] = host
    return pd.DataFrame(data)


def _single_col_df(seqs: list[str | None], col: str = "pb2_aa") -> pd.DataFrame:
    return pd.DataFrame({col: seqs})


AA_SEQS = ["MKTLL", "MKTLL", "MKTLL"]


def _df_seqs(seqs: list, col: str = "HA_aa") -> pd.DataFrame:
    return pd.DataFrame({col: seqs})


# ---------------------------------------------------------------------------
# sequences_to_label_encoding
# ---------------------------------------------------------------------------


class TestSequencesToLabelEncoding:
    def test_shape(self) -> None:
        df = _df_seqs(AA_SEQS)
        result = sequences_to_label_encoding(
            df, "HA_aa", length=10, alphabet=AA_ALPHABET
        )
        assert result.shape == (3, 10)

    def test_dtype(self) -> None:
        df = _df_seqs(AA_SEQS)
        result = sequences_to_label_encoding(
            df, "HA_aa", length=5, alphabet=AA_ALPHABET
        )
        assert result.dtype == np.int16

    def test_known_chars_nonzero(self) -> None:
        df = _df_seqs(["M"])
        result = sequences_to_label_encoding(
            df, "HA_aa", length=5, alphabet=AA_ALPHABET
        )
        # 'M' is at index 9 in AA_ALPHABET (0-based) -> encoded as 10 (1-based)
        assert result[0, 0] == AA_ALPHABET.index("M") + 1

    def test_padding_is_zero(self) -> None:
        df = _df_seqs(["M"])  # length 1, padded to 5
        result = sequences_to_label_encoding(
            df, "HA_aa", length=5, alphabet=AA_ALPHABET
        )
        assert result[0, 1] == 0
        assert result[0, 4] == 0

    def test_truncation(self) -> None:
        df = _df_seqs(["MKTLL"])  # length 5, truncated to 3
        result = sequences_to_label_encoding(
            df, "HA_aa", length=3, alphabet=AA_ALPHABET
        )
        assert result.shape[1] == 3

    def test_unknown_char_is_zero(self) -> None:
        df = _df_seqs(["MX"])  # 'X' not in AA_ALPHABET
        result = sequences_to_label_encoding(
            df, "HA_aa", length=5, alphabet=AA_ALPHABET
        )
        assert result[0, 1] == 0

    def test_nan_row_is_zeros(self) -> None:
        df = _df_seqs([None])
        result = sequences_to_label_encoding(
            df, "HA_aa", length=5, alphabet=AA_ALPHABET
        )
        assert (result[0] == 0).all()


# ---------------------------------------------------------------------------
# sequences_to_ohe_tensor
# ---------------------------------------------------------------------------


class TestSequencesToOheTensor:
    def test_shape(self) -> None:
        df = _df_seqs(AA_SEQS)
        result = sequences_to_ohe_tensor(df, "HA_aa", length=5, alphabet=AA_ALPHABET)
        assert result.shape == (3, 5, len(AA_ALPHABET))

    def test_dtype(self) -> None:
        df = _df_seqs(AA_SEQS)
        result = sequences_to_ohe_tensor(df, "HA_aa", length=5, alphabet=AA_ALPHABET)
        assert result.dtype == np.float32

    def test_known_position_is_one_hot(self) -> None:
        df = _df_seqs(["M"])
        result = sequences_to_ohe_tensor(df, "HA_aa", length=5, alphabet=AA_ALPHABET)
        idx = AA_ALPHABET.index("M")
        assert result[0, 0, idx] == 1.0
        assert result[0, 0].sum() == 1.0

    def test_padding_positions_all_zero(self) -> None:
        df = _df_seqs(["M"])  # length 1, padded to 5
        result = sequences_to_ohe_tensor(df, "HA_aa", length=5, alphabet=AA_ALPHABET)
        assert result[0, 1].sum() == 0.0
        assert result[0, 4].sum() == 0.0

    def test_truncation(self) -> None:
        df = _df_seqs(["MKTLL"])
        result = sequences_to_ohe_tensor(df, "HA_aa", length=3, alphabet=AA_ALPHABET)
        assert result.shape == (1, 3, len(AA_ALPHABET))

    def test_unknown_char_all_zero(self) -> None:
        df = _df_seqs(["MX"])
        result = sequences_to_ohe_tensor(df, "HA_aa", length=5, alphabet=AA_ALPHABET)
        assert result[0, 1].sum() == 0.0

    def test_nan_row_all_zero(self) -> None:
        df = _df_seqs([None])
        result = sequences_to_ohe_tensor(df, "HA_aa", length=5, alphabet=AA_ALPHABET)
        assert result[0].sum() == 0.0

    def test_flatten_for_linear_model(self) -> None:
        df = _df_seqs(AA_SEQS)
        result = sequences_to_ohe_tensor(df, "HA_aa", length=5, alphabet=AA_ALPHABET)
        flat = result.reshape(len(df), -1)
        assert flat.shape == (3, 5 * len(AA_ALPHABET))


# ---------------------------------------------------------------------------
# _segment_name helper
# ---------------------------------------------------------------------------


def test_segment_name_strips_aa_suffix() -> None:
    assert _segment_name("pb2_aa") == "pb2"
    assert _segment_name("ha_aa") == "ha"
    assert _segment_name("pb1_f2_aa") == "pb1_f2"
    assert _segment_name("pa_x_aa") == "pa_x"


def test_segment_name_no_suffix_unchanged() -> None:
    assert _segment_name("pb2") == "pb2"


# ---------------------------------------------------------------------------
# PositionalOneHotEncoder: basic fit / transform
# ---------------------------------------------------------------------------


class TestPositionalOneHotEncoderBasic:
    def test_fit_returns_self(self) -> None:
        df = _single_col_df(SEQ_PB2)
        enc = PositionalOneHotEncoder()
        result = enc.fit(df)
        assert result is enc

    def test_fit_sets_columns(self) -> None:
        df = _single_col_df(SEQ_PB2)
        enc = PositionalOneHotEncoder()
        enc.fit(df)
        assert enc.columns_ == ["pb2_aa"]

    def test_fit_sets_seq_lengths(self) -> None:
        df = _single_col_df(SEQ_PB2)
        enc = PositionalOneHotEncoder()
        enc.fit(df)
        assert enc.seq_lengths_["pb2_aa"] == PB2_LEN

    def test_transform_returns_dataframe(self) -> None:
        df = _single_col_df(SEQ_PB2)
        enc = PositionalOneHotEncoder()
        enc.fit(df)
        result = enc.transform(df)
        assert isinstance(result, pd.DataFrame)

    def test_transform_row_count(self) -> None:
        df = _single_col_df(SEQ_PB2)
        enc = PositionalOneHotEncoder()
        X = enc.fit_transform(df)
        assert len(X) == len(SEQ_PB2)

    def test_transform_preserves_index(self) -> None:
        df = _single_col_df(SEQ_PB2).set_index(pd.Index(["a", "b", "c"]))
        enc = PositionalOneHotEncoder()
        X = enc.fit_transform(df)
        assert list(X.index) == ["a", "b", "c"]

    def test_transform_requires_fit_first(self) -> None:
        df = _single_col_df(SEQ_PB2)
        enc = PositionalOneHotEncoder()
        with pytest.raises(RuntimeError, match="fit"):
            enc.transform(df)


# ---------------------------------------------------------------------------
# Feature naming
# ---------------------------------------------------------------------------


class TestFeatureNaming:
    def test_feature_name_format(self) -> None:
        """Feature names must follow {segment}_{pos}{residue} format."""
        df = _single_col_df(["MKTLL"])
        enc = PositionalOneHotEncoder(shrink=True)
        enc.fit(df)
        # All sequences are identical → each position has exactly one residue
        assert "pb2_1M" in enc.feature_names_out_
        assert "pb2_2K" in enc.feature_names_out_
        assert "pb2_3T" in enc.feature_names_out_
        assert "pb2_4L" in enc.feature_names_out_
        assert "pb2_5L" in enc.feature_names_out_

    def test_feature_names_match_columns(self) -> None:
        df = _single_col_df(SEQ_PB2)
        enc = PositionalOneHotEncoder(shrink=True)
        X = enc.fit_transform(df)
        assert list(X.columns) == enc.feature_names_out_

    def test_multi_col_feature_names_include_segment(self) -> None:
        df = _df(pb2=SEQ_PB2, ha=SEQ_HA)
        enc = PositionalOneHotEncoder(shrink=True)
        enc.fit(df)
        names = enc.feature_names_out_
        # pb2 features come first
        pb2_names = [n for n in names if n.startswith("pb2_")]
        ha_names = [n for n in names if n.startswith("ha_")]
        assert len(pb2_names) > 0
        assert len(ha_names) > 0
        # Order: pb2 columns first
        first_ha_idx = names.index(ha_names[0])
        last_pb2_idx = names.index(pb2_names[-1])
        assert last_pb2_idx < first_ha_idx

    def test_position_is_one_indexed(self) -> None:
        """Positions in feature names are 1-based."""
        df = _single_col_df(["M"])  # length 1
        enc = PositionalOneHotEncoder(shrink=True)
        enc.fit(df)
        # Should have pb2_1M, not pb2_0M
        assert any(n.startswith("pb2_1") for n in enc.feature_names_out_)
        assert not any(n.startswith("pb2_0") for n in enc.feature_names_out_)

    def test_non_aa_col_name_uses_full_name_as_segment(self) -> None:
        df = pd.DataFrame({"myseq": ["MKTLL", "MKTLL"]})
        enc = PositionalOneHotEncoder(aa_columns=["myseq"], shrink=True)
        enc.fit(df)
        assert all(n.startswith("myseq_") for n in enc.feature_names_out_)


# ---------------------------------------------------------------------------
# One-hot encoding correctness
# ---------------------------------------------------------------------------


class TestEncoding:
    def test_single_identical_seqs_each_pos_sums_to_one(self) -> None:
        """With shrink and identical sequences, each row has 1 per position."""
        df = _single_col_df(["MKTLL", "MKTLL", "MKTLL"])
        enc = PositionalOneHotEncoder(shrink=True)
        X = enc.fit_transform(df)
        # Each row should sum to sequence length (one 1 per position)
        assert (X.sum(axis=1) == PB2_LEN).all()

    def test_correct_value_at_known_position(self) -> None:
        """pb2_1M should be 1 for 'MKTLL' and 0 for 'AKTLL'."""
        df = _single_col_df(["MKTLL", "AKTLL"], col="pb2_aa")
        enc = PositionalOneHotEncoder(shrink=False, alphabet="ACDEFGHIKLMNPQRSTVWY")
        X = enc.fit_transform(df)
        assert X["pb2_1M"].iloc[0] == 1
        assert X["pb2_1M"].iloc[1] == 0
        assert X["pb2_1A"].iloc[0] == 0
        assert X["pb2_1A"].iloc[1] == 1

    def test_nan_row_all_zero(self) -> None:
        df = _single_col_df([None, "MKTLL"])
        enc = PositionalOneHotEncoder(shrink=True)
        X = enc.fit_transform(df)
        assert X.iloc[0].sum() == 0

    def test_dtype_is_uint8(self) -> None:
        df = _single_col_df(SEQ_PB2)
        enc = PositionalOneHotEncoder(shrink=True)
        X = enc.fit_transform(df)
        assert X.dtypes.unique()[0] == np.uint8

    def test_values_are_binary(self) -> None:
        df = _single_col_df(SEQ_PB2)
        enc = PositionalOneHotEncoder()
        X = enc.fit_transform(df)
        assert set(X.values.ravel().tolist()).issubset({0, 1})


# ---------------------------------------------------------------------------
# Shrink option
# ---------------------------------------------------------------------------


class TestShrink:
    def test_shrink_reduces_feature_count(self) -> None:
        """Shrink should produce fewer features than no-shrink."""
        df = _single_col_df(["MKTLL", "MKTLL"])  # only M,K,T,L residues observed
        enc_shrink = PositionalOneHotEncoder(shrink=True, alphabet=AA_ALPHABET)
        enc_full = PositionalOneHotEncoder(shrink=False, alphabet=AA_ALPHABET)
        X_shrink = enc_shrink.fit_transform(df)
        X_full = enc_full.fit_transform(df)
        assert X_shrink.shape[1] < X_full.shape[1]

    def test_shrink_no_all_zero_columns(self) -> None:
        """After shrink, no column should be all zeros."""
        df = _single_col_df(SEQ_PB2)
        enc = PositionalOneHotEncoder(shrink=True, alphabet=AA_ALPHABET)
        X = enc.fit_transform(df)
        assert (X.sum(axis=0) > 0).all()

    def test_no_shrink_may_have_zero_columns(self) -> None:
        """Without shrink, some columns will be all zeros (unobserved residues)."""
        # 'MKTLL' only uses M, K, T, L; with full AA alphabet many are zero
        df = _single_col_df(["MKTLL", "MKTLL"])
        enc = PositionalOneHotEncoder(shrink=False, alphabet=AA_ALPHABET)
        X = enc.fit_transform(df)
        zero_cols = (X.sum(axis=0) == 0).sum()
        assert zero_cols > 0

    def test_shrink_drops_star_when_absent(self) -> None:
        """Stop codon '*' feature should not appear when not in training data."""
        df = _single_col_df(["MKTLL", "MKTLL"])
        enc = PositionalOneHotEncoder(shrink=True, alphabet=AA_ALPHABET_EXTENDED)
        enc.fit(df)
        star_features = [n for n in enc.feature_names_out_ if n.endswith("*")]
        assert len(star_features) == 0

    def test_shrink_keeps_star_when_present(self) -> None:
        """Stop codon '*' feature must be kept if observed in training data."""
        df = _single_col_df(["MK*LL", "MKTLL"])
        enc = PositionalOneHotEncoder(shrink=True, alphabet=AA_ALPHABET_EXTENDED)
        enc.fit(df)
        star_features = [n for n in enc.feature_names_out_ if n.endswith("*")]
        assert len(star_features) > 0

    def test_transform_on_new_data_uses_fitted_vocab(self) -> None:
        """Residues unseen during fit should not appear in transform output."""
        train = _single_col_df(["MKTLL", "MKTLL"])
        test = _single_col_df(["ACDEF"])  # entirely different residues
        enc = PositionalOneHotEncoder(shrink=True, alphabet=AA_ALPHABET)
        enc.fit(train)
        X_test = enc.transform(test)
        # Features are fixed to training vocab; test rows will be mostly zeros
        assert list(X_test.columns) == enc.feature_names_out_
        # None of the training residues (M,K,T,L) appear in test → all zeros
        assert X_test.sum(axis=1).iloc[0] == 0


# ---------------------------------------------------------------------------
# Multiple columns
# ---------------------------------------------------------------------------


class TestMultipleColumns:
    def test_multi_col_shape(self) -> None:
        df = _df(pb2=SEQ_PB2, ha=SEQ_HA)
        enc = PositionalOneHotEncoder(shrink=True)
        X = enc.fit_transform(df)
        pb2_feats = sum(1 for n in enc.feature_names_out_ if n.startswith("pb2_"))
        ha_feats = sum(1 for n in enc.feature_names_out_ if n.startswith("ha_"))
        assert X.shape[1] == pb2_feats + ha_feats

    def test_explicit_aa_columns(self) -> None:
        df = _df(pb2=SEQ_PB2, ha=SEQ_HA)
        enc = PositionalOneHotEncoder(aa_columns=["pb2_aa"], shrink=True)
        X = enc.fit_transform(df)
        assert all(n.startswith("pb2_") for n in X.columns)

    def test_missing_column_raises(self) -> None:
        df = pd.DataFrame({"other_col": ["MKTLL"]})
        enc = PositionalOneHotEncoder(aa_columns=["pb2_aa"])
        with pytest.raises(ValueError, match="pb2_aa"):
            enc.fit(df)


# ---------------------------------------------------------------------------
# get_feature_names_out / repr
# ---------------------------------------------------------------------------


class TestMeta:
    def test_get_feature_names_out(self) -> None:
        df = _single_col_df(SEQ_PB2)
        enc = PositionalOneHotEncoder(shrink=True)
        enc.fit(df)
        assert enc.get_feature_names_out() == enc.feature_names_out_

    def test_repr_contains_key_info(self) -> None:
        enc = PositionalOneHotEncoder(shrink=True)
        r = repr(enc)
        assert "shrink=True" in r
        assert "not fitted" in r

    def test_repr_after_fit(self) -> None:
        df = _single_col_df(SEQ_PB2)
        enc = PositionalOneHotEncoder(shrink=True)
        enc.fit(df)
        r = repr(enc)
        assert "fitted" in r


# ---------------------------------------------------------------------------
# sklearn compatibility smoke test
# ---------------------------------------------------------------------------


class TestSklearnCompatibility:
    def test_random_forest_accepts_output(self) -> None:
        pytest.importorskip("sklearn")
        from sklearn.ensemble import RandomForestClassifier

        seqs_human = ["MKTLL", "MKTLV", "MKTLK", "MKTLA", "MKTLS"]
        seqs_swine = ["AKTLL", "AKTLV", "AKTLK", "AKTLA", "AKTLS"]
        df = pd.DataFrame(
            {
                "pb2_aa": seqs_human + seqs_swine,
                "host": ["human"] * 5 + ["swine"] * 5,
            }
        )
        enc = PositionalOneHotEncoder(shrink=True)
        X = enc.fit_transform(df)
        y = df["host"]
        clf = RandomForestClassifier(n_estimators=10, random_state=0)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert len(preds) == len(df)

    def test_feature_importances_align_with_feature_names(self) -> None:
        pytest.importorskip("sklearn")
        from sklearn.ensemble import RandomForestClassifier

        seqs_human = ["MKTLL", "MKTLV", "MKTLK"]
        seqs_swine = ["AKTLL", "AKTLV", "AKTLK"]
        df = pd.DataFrame(
            {
                "pb2_aa": seqs_human + seqs_swine,
                "host": ["human"] * 3 + ["swine"] * 3,
            }
        )
        enc = PositionalOneHotEncoder(shrink=True)
        X = enc.fit_transform(df)
        clf = RandomForestClassifier(n_estimators=10, random_state=0)
        clf.fit(X, df["host"])
        assert len(clf.feature_importances_) == len(enc.feature_names_out_)
