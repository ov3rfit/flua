"""Tests for quality flags on AnalyzedSequence and GeneProduct."""

from __future__ import annotations

import pytest
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from flua.models import AnalyzedSequence
from flua.products import GeneProduct


def _make_seq(aa_seq: str | None) -> AnalyzedSequence:
    return AnalyzedSequence(
        record=SeqRecord(Seq("ATG")),
        seq_type="DNA",
        aa_seq=aa_seq,
        segment_name="HA",
    )


def _make_product(aa_seq: str) -> GeneProduct:
    return GeneProduct(
        name="M2",
        mechanism="splicing",
        nt_seq="ATG",
        aa_seq=aa_seq,
    )


class TestAnalyzedSequenceFlags:
    def test_no_flags_on_normal_sequence(self) -> None:
        seq = _make_seq("MKTLL")
        assert not seq.has_stop_codon
        assert not seq.has_ambiguous

    def test_has_stop_codon(self) -> None:
        seq = _make_seq("MKT*LL")
        assert seq.has_stop_codon
        assert not seq.has_ambiguous

    def test_has_ambiguous(self) -> None:
        seq = _make_seq("MKXLL")
        assert not seq.has_stop_codon
        assert seq.has_ambiguous

    def test_both_flags(self) -> None:
        seq = _make_seq("MK*XLL")
        assert seq.has_stop_codon
        assert seq.has_ambiguous

    def test_none_aa_seq_has_no_flags(self) -> None:
        seq = _make_seq(None)
        assert not seq.has_stop_codon
        assert not seq.has_ambiguous


class TestGeneProductFlags:
    def test_no_flags_on_normal_product(self) -> None:
        prod = _make_product("MKTLL")
        assert not prod.has_stop_codon
        assert not prod.has_ambiguous

    def test_has_stop_codon(self) -> None:
        prod = _make_product("MKT*")
        assert prod.has_stop_codon
        assert not prod.has_ambiguous

    def test_has_ambiguous(self) -> None:
        prod = _make_product("MKXLL")
        assert not prod.has_stop_codon
        assert prod.has_ambiguous
