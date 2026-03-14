"""Tests for flua.io: FASTA loading and DataFrame conversion."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from flua.io import groups_to_dataframe, load_fasta, load_multiple_fasta


class TestLoadFasta:
    def test_loads_all_segments(self, h1n1_fasta: Path) -> None:
        group = load_fasta(h1n1_fasta)
        assert len(group.sequences) == 8

    def test_detects_subtype(self, h1n1_fasta: Path) -> None:
        group = load_fasta(h1n1_fasta)
        assert group.subtype == "H1N1"

    def test_group_name_defaults_to_stem(self, h1n1_fasta: Path) -> None:
        group = load_fasta(h1n1_fasta)
        assert group.group_name == "test_h1n1"

    def test_custom_group_name(self, h1n1_fasta: Path) -> None:
        group = load_fasta(h1n1_fasta, group_name="custom")
        assert group.group_name == "custom"

    def test_no_subtype_detected(self, unknown_fasta: Path) -> None:
        group = load_fasta(unknown_fasta)
        assert group.subtype is None

    def test_identifies_segments(self, h1n1_fasta: Path) -> None:
        group = load_fasta(h1n1_fasta)
        names = [s.segment_name for s in group.sequences]
        assert "PB2" in names
        assert "NS" in names

    def test_generates_alternative_products(self, h1n1_fasta: Path) -> None:
        group = load_fasta(h1n1_fasta)
        all_products = group.get_all_products()
        product_names = {p.name for _, p in all_products}
        # At minimum, direct-translation products should exist for each
        # segment.
        assert "PB2" in product_names
        assert "M1" in product_names


class TestLoadMultipleFasta:
    def test_returns_list_of_groups(self, h1n1_fasta: Path, h5n1_fasta: Path) -> None:
        groups = load_multiple_fasta([h1n1_fasta, h5n1_fasta])
        assert len(groups) == 2

    def test_subtypes(self, h1n1_fasta: Path, h5n1_fasta: Path) -> None:
        groups = load_multiple_fasta([h1n1_fasta, h5n1_fasta])
        subtypes = {g.subtype for g in groups}
        assert subtypes == {"H1N1", "H5N1"}


class TestGroupsToDataframe:
    def test_dataframe_shape(
        self,
        h1n1_fasta: Path,
        h5n1_fasta: Path,
        unknown_fasta: Path,
    ) -> None:
        groups = load_multiple_fasta([h1n1_fasta, h5n1_fasta, unknown_fasta])
        df = groups_to_dataframe(groups, value_type="translated")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_subtype_column(
        self,
        h1n1_fasta: Path,
        unknown_fasta: Path,
    ) -> None:
        groups = load_multiple_fasta([h1n1_fasta, unknown_fasta])
        df = groups_to_dataframe(groups)
        assert "subtype" in df.columns
        assert df.iloc[0]["subtype"] == "H1N1"
        assert pd.isna(df.iloc[1]["subtype"])

    def test_subtype_filtering(
        self,
        h1n1_fasta: Path,
        h5n1_fasta: Path,
    ) -> None:
        groups = load_multiple_fasta([h1n1_fasta, h5n1_fasta])
        df = groups_to_dataframe(groups)
        h1n1_df = df[df["subtype"] == "H1N1"]
        assert len(h1n1_df) == 1

    def test_raw_columns_use_raw_suffix(
        self,
        h1n1_fasta: Path,
    ) -> None:
        groups = load_multiple_fasta([h1n1_fasta])
        df = groups_to_dataframe(groups, value_type="raw")
        assert "PA_raw" in df.columns
        assert "PA_type" in df.columns
        # No legacy column names
        assert "PA_seq" not in df.columns
        assert "PA_length" not in df.columns

    def test_translated_columns_use_aa_suffix(
        self,
        h1n1_fasta: Path,
    ) -> None:
        groups = load_multiple_fasta([h1n1_fasta])
        df = groups_to_dataframe(groups, value_type="translated")
        assert "PA_aa" in df.columns
        assert "PA_type" in df.columns
        # No legacy column names
        assert "PA_seq" not in df.columns
        assert "PA_protein" not in df.columns

    def test_no_length_columns(
        self,
        h1n1_fasta: Path,
    ) -> None:
        groups = load_multiple_fasta([h1n1_fasta])
        df = groups_to_dataframe(groups, value_type="translated")
        length_cols = [c for c in df.columns if "_length" in c]
        assert length_cols == []

    def test_no_direct_product_duplication(
        self,
        h1n1_fasta: Path,
    ) -> None:
        """Direct-mechanism products should not create separate columns
        since they duplicate the segment column."""
        groups = load_multiple_fasta([h1n1_fasta])
        df = groups_to_dataframe(groups, value_type="translated")
        # PB2 is a direct product — should NOT appear as PB2_protein or PB2_aa
        # alongside the segment column PB2_aa
        col_names = list(df.columns)
        pb2_cols = [c for c in col_names if c.startswith("PB2_") and c != "PB2_aa" and c != "PB2_type"]
        assert pb2_cols == [], f"Unexpected PB2 duplicate columns: {pb2_cols}"

    def test_alt_product_columns_present(
        self,
        h1n1_fasta: Path,
    ) -> None:
        groups = load_multiple_fasta([h1n1_fasta])
        df = groups_to_dataframe(groups, value_type="translated")
        alt_cols = [c for c in df.columns if c.endswith("_aa") and c.split("_")[0] not in
                    ["PB2", "PB1", "PA", "HA", "NP", "NA", "MP", "NS"]]
        assert len(alt_cols) > 0

    def test_alt_product_columns_absent_when_disabled(
        self,
        h1n1_fasta: Path,
    ) -> None:
        groups = load_multiple_fasta([h1n1_fasta])
        df = groups_to_dataframe(groups, value_type="translated", include_alt_products=False)
        # Only segment _aa and _type columns should exist (plus metadata)
        non_meta = [c for c in df.columns if c not in
                    ("group_name", "source_file", "subtype", "num_sequences")]
        for col in non_meta:
            assert col.endswith("_aa") or col.endswith("_type")
            seg = col.rsplit("_", 1)[0]
            assert seg in ["PB2", "PB1", "PA", "HA", "NP", "NA", "MP", "NS"]
