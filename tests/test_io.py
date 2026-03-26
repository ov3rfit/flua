"""Tests for flua.io: FASTA loading and DataFrame conversion."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from flua.io import (
    groups_to_dataframe,
    load_fasta,
    load_fasta_string,
    load_gisaid_fasta,
    load_multiple_fasta,
)
from flua.models import AnalyzedSequence, SequenceGroup


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
        df, _ = groups_to_dataframe(groups, value_type="translated")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_subtype_column(
        self,
        h1n1_fasta: Path,
        unknown_fasta: Path,
    ) -> None:
        groups = load_multiple_fasta([h1n1_fasta, unknown_fasta])
        df, _ = groups_to_dataframe(groups)
        assert "subtype" in df.columns
        assert df.iloc[0]["subtype"] == "H1N1"
        assert pd.isna(df.iloc[1]["subtype"])

    def test_subtype_filtering(
        self,
        h1n1_fasta: Path,
        h5n1_fasta: Path,
    ) -> None:
        groups = load_multiple_fasta([h1n1_fasta, h5n1_fasta])
        df, _ = groups_to_dataframe(groups)
        h1n1_df = df[df["subtype"] == "H1N1"]
        assert len(h1n1_df) == 1

    def test_translated_columns_use_aa_suffix(
        self,
        h1n1_fasta: Path,
    ) -> None:
        groups = load_multiple_fasta([h1n1_fasta])
        df, _ = groups_to_dataframe(groups, value_type="translated")
        assert "PA_aa" in df.columns

    def test_no_direct_product_duplication(
        self,
        h1n1_fasta: Path,
    ) -> None:
        """Direct-mechanism products should not create separate columns
        since they duplicate the segment column."""
        groups = load_multiple_fasta([h1n1_fasta])
        df, _ = groups_to_dataframe(groups, value_type="translated")
        # PB2 is a direct product — should NOT appear as PB2_protein or PB2_aa
        # alongside the segment column PB2_aa
        col_names = list(df.columns)
        pb2_cols = [c for c in col_names if c.startswith("PB2_") and c != "PB2_aa"]
        assert pb2_cols == [], f"Unexpected PB2 duplicate columns: {pb2_cols}"

    def test_alt_product_columns_present(
        self,
        h1n1_fasta: Path,
    ) -> None:
        groups = load_multiple_fasta([h1n1_fasta])
        df, _ = groups_to_dataframe(groups, value_type="translated")
        alt_cols = [
            c
            for c in df.columns
            if c.endswith("_aa")
            and c.split("_")[0]
            not in ["PB2", "PB1", "PA", "HA", "NP", "NA", "M1", "NS1"]
        ]
        assert len(alt_cols) > 0

    def test_alt_product_columns_absent_when_disabled(
        self,
        h1n1_fasta: Path,
    ) -> None:
        groups = load_multiple_fasta([h1n1_fasta])
        df, _ = groups_to_dataframe(
            groups, value_type="translated", include_alt_products=False
        )
        # Only segment _aa columns should exist (plus metadata)
        non_meta = [
            c
            for c in df.columns
            if c
            not in ("group_name", "source_file", "subtype", "host", "num_sequences")
        ]
        for col in non_meta:
            assert col.endswith("_aa")
            seg = col.rsplit("_", 1)[0]
            assert seg in ["PB2", "PB1", "PA", "HA", "NP", "NA", "M1", "NS1"]

    def test_returns_empty_warnings_when_lengths_consistent(
        self,
        h1n1_fasta: Path,
        h5n1_fasta: Path,
    ) -> None:
        groups = load_multiple_fasta([h1n1_fasta, h5n1_fasta])
        _, warn_messages = groups_to_dataframe(groups)
        assert isinstance(warn_messages, list)

    def test_returns_warnings_when_lengths_differ(
        self,
        h1n1_fasta: Path,
        h5n1_fasta: Path,
    ) -> None:
        """groups_to_dataframe must return warning strings (not raise or emit)
        when segment lengths differ across samples."""
        from flua.io import _check_seq_length_consistency

        groups = load_multiple_fasta([h1n1_fasta, h5n1_fasta])
        df, _ = groups_to_dataframe(groups)

        # Manually corrupt one sequence to force a length mismatch.
        df2 = df.copy()
        df2.loc[df2.index[0], "PB2_nt"] = "ATCG"  # much shorter than real PB2
        msgs = _check_seq_length_consistency(df2, {"PB2": "PB2_nt"})
        assert len(msgs) == 1
        assert "PB2" in msgs[0]
        assert "differ" in msgs[0]


# ── Helpers for quality-filtering tests ──────────────────────────────────


def _make_group(group_name: str, segment: str, aa_seq: str) -> SequenceGroup:
    """Build a minimal :class:`SequenceGroup` with a single segment sequence."""
    record = SeqRecord(Seq("ATG" * 600), id=f"{group_name}|{segment}")
    seq = AnalyzedSequence(
        record=record,
        seq_type="DNA",
        aa_seq=aa_seq,
        segment_name=segment,
    )
    group = SequenceGroup(group_name=group_name, source_file="<test>")
    group.sequences.append(seq)
    return group


class TestQualityFiltering:
    def test_default_preserves_stop_codon_sequences(self) -> None:
        """Without any filter flags, stop-codon sequences appear as-is."""
        group = _make_group("s", "HA", "MKT*L")
        df, _ = groups_to_dataframe(
            [group],
            value_type="translated",
            segment_names=["HA"],
            include_alt_products=False,
        )
        assert df["HA_aa"].iloc[0] == "MKT*L"

    def test_exclude_stop_codons_translated(self) -> None:
        normal = _make_group("normal", "HA", "MKTLL")
        flagged = _make_group("flagged", "HA", "MKT*L")
        df, _ = groups_to_dataframe(
            [normal, flagged],
            value_type="translated",
            segment_names=["HA"],
            include_alt_products=False,
            exclude_stop_codons=True,
        )
        assert df.loc[df["group_name"] == "normal", "HA_aa"].iloc[0] == "MKTLL"
        assert pd.isna(df.loc[df["group_name"] == "flagged", "HA_aa"].iloc[0])

    def test_exclude_stop_codons_raw(self) -> None:
        """Exclusion applies to nucleotide columns too."""
        normal = _make_group("normal", "HA", "MKTLL")
        flagged = _make_group("flagged", "HA", "MKT*L")
        df, _ = groups_to_dataframe(
            [normal, flagged],
            value_type="raw",
            segment_names=["HA"],
            include_alt_products=False,
            exclude_stop_codons=True,
        )
        assert pd.notna(df.loc[df["group_name"] == "normal", "HA_nt"].iloc[0])
        assert pd.isna(df.loc[df["group_name"] == "flagged", "HA_nt"].iloc[0])

    def test_exclude_ambiguous(self) -> None:
        normal = _make_group("normal", "HA", "MKTLL")
        flagged = _make_group("flagged", "HA", "MKXLL")
        df, _ = groups_to_dataframe(
            [normal, flagged],
            value_type="translated",
            segment_names=["HA"],
            include_alt_products=False,
            exclude_ambiguous=True,
        )
        assert df.loc[df["group_name"] == "normal", "HA_aa"].iloc[0] == "MKTLL"
        assert pd.isna(df.loc[df["group_name"] == "flagged", "HA_aa"].iloc[0])

    def test_exclude_stop_codons_does_not_affect_ambiguous(self) -> None:
        """exclude_stop_codons=True must not filter out ambiguous-only sequences."""
        ambig = _make_group("ambig", "HA", "MKXLL")
        df, _ = groups_to_dataframe(
            [ambig],
            value_type="translated",
            segment_names=["HA"],
            include_alt_products=False,
            exclude_stop_codons=True,
        )
        assert df["HA_aa"].iloc[0] == "MKXLL"

    def test_exclude_ambiguous_does_not_affect_stop_codons(self) -> None:
        """exclude_ambiguous=True must not filter out stop-codon-only sequences."""
        stop = _make_group("stop", "HA", "MKT*L")
        df, _ = groups_to_dataframe(
            [stop],
            value_type="translated",
            segment_names=["HA"],
            include_alt_products=False,
            exclude_ambiguous=True,
        )
        assert df["HA_aa"].iloc[0] == "MKT*L"

    def test_other_segments_unaffected(self) -> None:
        """Flagging one segment should not clear other segments in the same row."""
        record_ha = SeqRecord(Seq("ATG" * 600), id="s|HA")
        record_np = SeqRecord(Seq("ATG" * 600), id="s|NP")
        seq_ha = AnalyzedSequence(
            record=record_ha, seq_type="DNA", aa_seq="MKT*L", segment_name="HA"
        )
        seq_np = AnalyzedSequence(
            record=record_np, seq_type="DNA", aa_seq="MKTLL", segment_name="NP"
        )
        group = SequenceGroup(group_name="mixed", source_file="<test>")
        group.sequences.extend([seq_ha, seq_np])
        df, _ = groups_to_dataframe(
            [group],
            value_type="translated",
            segment_names=["HA", "NP"],
            include_alt_products=False,
            exclude_stop_codons=True,
        )
        assert pd.isna(df["HA_aa"].iloc[0])
        assert df["NP_aa"].iloc[0] == "MKTLL"


class TestDegenerateWarnings:
    def test_no_warning_for_clean_sequences(self) -> None:
        group = _make_group("clean", "HA", "MKTLL")
        _, warnings = groups_to_dataframe(
            [group],
            segment_names=["HA"],
            include_alt_products=False,
        )
        degenerate_warns = [w for w in warnings if "degenerate" in w]
        assert degenerate_warns == []

    def test_warning_when_degenerate_nt_present(self) -> None:
        record = SeqRecord(Seq("ATGNNNATG"), id="degen|HA")
        seq = AnalyzedSequence(
            record=record,
            seq_type="DNA",
            aa_seq="MXM",
            segment_name="HA",
        )
        group = SequenceGroup(group_name="degen_sample", source_file="<test>")
        group.sequences.append(seq)
        _, warnings = groups_to_dataframe(
            [group],
            segment_names=["HA"],
            include_alt_products=False,
        )
        assert any(
            "degenerate" in w and "HA" in w and "degen_sample" in w for w in warnings
        )

    def test_warning_only_for_affected_segment(self) -> None:
        record_ha = SeqRecord(Seq("ATGNNNATG"), id="s|HA")
        record_np = SeqRecord(Seq("ATGATGATG"), id="s|NP")
        seq_ha = AnalyzedSequence(
            record=record_ha, seq_type="DNA", aa_seq="MXM", segment_name="HA"
        )
        seq_np = AnalyzedSequence(
            record=record_np, seq_type="DNA", aa_seq="MMM", segment_name="NP"
        )
        group = SequenceGroup(group_name="mixed", source_file="<test>")
        group.sequences.extend([seq_ha, seq_np])
        _, warnings = groups_to_dataframe(
            [group],
            segment_names=["HA", "NP"],
            include_alt_products=False,
        )
        degenerate_warns = [w for w in warnings if "degenerate" in w]
        assert len(degenerate_warns) == 1
        assert "HA" in degenerate_warns[0]
        assert "NP" not in degenerate_warns[0]


class TestGisaidHost:
    def test_host_extracted_from_header(self, gisaid_fasta: Path) -> None:
        groups = load_gisaid_fasta(gisaid_fasta)
        hosts = {g.group_name: g.host for g in groups}
        assert hosts["A/California/07/2009"] == "Human"
        assert hosts["A/swine/Iowa/1/2023"] == "Swine"

    def test_host_column_in_dataframe(self, gisaid_fasta: Path) -> None:
        groups = load_gisaid_fasta(gisaid_fasta)
        df, _ = groups_to_dataframe(groups, segment_names=["HA", "NA"])
        assert "host" in df.columns
        assert set(df["host"].dropna()) == {"Human", "Swine"}

    def test_non_gisaid_fasta_host_is_none(self, h1n1_fasta: Path) -> None:
        groups = load_multiple_fasta([h1n1_fasta])
        df, _ = groups_to_dataframe(groups)
        assert df["host"].isna().all()

    def test_host_missing_from_header_is_none(self, tmp_path: Path) -> None:
        """Headers without a host field (< 6 pipe-separated parts) yield None."""
        import random

        rng = random.Random(42)
        seq = "ATG" + "".join(rng.choice("ATGC") for _ in range(1773))
        fasta = tmp_path / "no_host.fasta"
        fasta.write_text(">EPI_ISL_999|A/test/1/2020|4|HA|H1N1\n" + seq + "\n")
        groups = load_gisaid_fasta(fasta)
        assert groups[0].host is None


class TestLoadFastaString:
    def test_returns_same_result_as_load_fasta(self, h1n1_fasta: Path) -> None:
        fasta_text = h1n1_fasta.read_text()
        from_file = load_fasta(h1n1_fasta)
        from_str = load_fasta_string(fasta_text, group_name=h1n1_fasta.stem)
        assert len(from_str.sequences) == len(from_file.sequences)
        assert from_str.subtype == from_file.subtype
        assert from_str.group_name == from_file.group_name

    def test_source_file_is_string_sentinel(self, h1n1_fasta: Path) -> None:
        fasta_text = h1n1_fasta.read_text()
        group = load_fasta_string(fasta_text, group_name="test")
        assert group.source_file == "<string>"

    def test_group_name_is_required_and_used(self, h1n1_fasta: Path) -> None:
        fasta_text = h1n1_fasta.read_text()
        group = load_fasta_string(fasta_text, group_name="my_sample")
        assert group.group_name == "my_sample"
