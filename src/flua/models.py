"""Core data models for analyzed sequences and sequence groups."""

from __future__ import annotations

from dataclasses import dataclass, field

from Bio.SeqRecord import SeqRecord

from flua.products import AlternativeProduct


@dataclass
class AnalyzedSequence:
    """A single analyzed sequence (typically one influenza segment)."""

    record: SeqRecord
    seq_type: str
    translated: str | None
    segment_name: str | None
    alt_products: list[AlternativeProduct] = field(default_factory=list)
    length: int = 0

    def __post_init__(self) -> None:
        self.length = len(self.record.seq)

    @property
    def id(self) -> str:
        return self.record.id

    @property
    def description(self) -> str:
        return self.record.description

    @property
    def raw_sequence(self) -> str:
        return str(self.record.seq)

    def get_product(self, name: str) -> AlternativeProduct | None:
        """Look up an alternative product by *name* (case-insensitive)."""
        for p in self.alt_products:
            if p.name.upper() == name.upper():
                return p
        return None


@dataclass
class SequenceGroup:
    """A collection of sequences originating from a single FASTA file."""

    group_name: str
    source_file: str
    subtype: str | None = None
    sequences: list[AnalyzedSequence] = field(default_factory=list)

    @property
    def segment_names(self) -> list[str | None]:
        return [s.segment_name for s in self.sequences]

    def get_segment(self, name: str) -> AnalyzedSequence | None:
        """Return the sequence whose segment name matches *name*
        (case-insensitive)."""
        for seq in self.sequences:
            if seq.segment_name and seq.segment_name.upper() == name.upper():
                return seq
        return None

    def get_all_products(self) -> list[tuple[str | None, AlternativeProduct]]:
        """Return ``(segment_name, product)`` pairs for every alternative
        product across all sequences in the group."""
        results = []
        for seq in self.sequences:
            for p in seq.alt_products:
                results.append((seq.segment_name, p))
        return results
