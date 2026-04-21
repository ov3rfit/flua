"""Tests for flua.seq_utils: subtype extraction and segment identification."""

from __future__ import annotations

import pytest

from flua.seq_utils import detect_sequence_type, extract_subtype, identify_segment

# ── Subtype extraction ───────────────────────────────────────────────────


class TestExtractSubtype:
    @pytest.mark.parametrize(
        ("header_id", "description", "expected"),
        [
            ("CY012345", "A/California/07/2009(H1N1)", "H1N1"),
            ("seq1", "A/duck/Vietnam/2005|H5N1|segment 3", "H5N1"),
            ("seq2|H3N2|sample", "Influenza A", "H3N2"),
            ("seq3", "A/Shanghai/2/2013 H7N9 PA", "H7N9"),
            ("seq4", "A/Indonesia/5/2005(H5N1)pdm09", "H5N1pdm09"),
            ("seq5", "H1N1pdm09 virus sample", "H1N1pdm09"),
            ("seq6", "Influenza A H5 subtype N6 virus", "H5N6"),
            ("seq7", "no subtype info here", None),
            ("seq8", "random sequence DYNAMIC", None),
            ("seq9", "A/swine/Iowa/2020 (H1N2)", "H1N2"),
            ("seq10", "H9N2 avian influenza", "H9N2"),
        ],
    )
    def test_subtype_extraction(
        self, header_id: str, description: str, expected: str | None
    ) -> None:
        assert extract_subtype(header_id, description) == expected


# ── Segment identification ───────────────────────────────────────────────


class TestIdentifySegment:
    @pytest.mark.parametrize(
        ("header_id", "description", "expected"),
        [
            ("seq1|PB2|sample", "Influenza A", "PB2"),
            ("seq|PA|sample", "PA subunit", "PA"),
            ("seq|PA-X|test", "PA-X protein", None),
            ("DYNAMIC_seq", "aeroDYNAMICS study", None),
            ("seg7_MP_flu", "Matrix protein", "MP"),
            # ``M`` resolves to the canonical ``MP`` segment name.
            ("seg7_M_flu", "Matrix protein", "MP"),
            ("sample|M|1", "segment 7", "MP"),
            ("sample_M", "Influenza A", "MP"),
            ("M_sample", "Influenza A", "MP"),
            # ``M1`` / ``M2`` are gene products, not the segment — must not
            # match as ``MP`` via the ``M`` alias.
            ("sample|M1|1", "M1 protein", None),
            ("sample|M2|1", "M2 protein", None),
            # ``m`` inside an unrelated word must not match as ``MP``.
            ("A/mallard/Alberta/1/2020", "H5N1", None),
        ],
    )
    def test_segment_matching(
        self, header_id: str, description: str, expected: str | None
    ) -> None:
        assert identify_segment(header_id, description) == expected


# ── Sequence type detection ──────────────────────────────────────────────


class TestDetectSequenceType:
    def test_dna(self) -> None:
        assert detect_sequence_type("ATGCGATCGATCG") == "DNA"

    def test_rna(self) -> None:
        assert detect_sequence_type("AUGCGAUCGAUCG") == "RNA"

    def test_protein(self) -> None:
        assert detect_sequence_type("MFLIVSPQR") == "Protein"

    def test_ambiguous_defaults_to_dna(self) -> None:
        # Only A, G, C – no distinguishing characters
        assert detect_sequence_type("AGCAGCAGC") == "DNA"

    def test_degenerate_dna_not_misclassified_as_protein(self) -> None:
        # IUPAC degenerate codes (R, S, W, K, M, H, V) overlap with
        # PROTEIN_ONLY_CHARS; high GCAT ratio (>90%) must override protein
        # detection.  20 GCAT + 2 degenerate = 20/22 ≈ 90.9%.
        assert detect_sequence_type("ATGCATGCATGCATGCATGCRS") == "DNA"

    def test_degenerate_dna_heavy_degenerate(self) -> None:
        # >90% GCAT → DNA even with degenerate codes present
        assert detect_sequence_type("ATGCATGCATGCATGCN") == "DNA"

    def test_degenerate_below_threshold_is_protein(self) -> None:
        # Low GCAT ratio with protein-only chars → Protein
        assert detect_sequence_type("MFLIVSPHQEDKWRV") == "Protein"
