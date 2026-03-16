"""Core data models for analyzed sequences and sequence groups."""

from __future__ import annotations

from dataclasses import dataclass, field

from Bio.SeqRecord import SeqRecord

from flua.products import GeneProduct


@dataclass
class AnalyzedSequence:
    """A single analyzed sequence (typically one influenza segment)."""

    record: SeqRecord
    seq_type: str
    aa_seq: str | None
    segment_name: str | None
    alt_products: list[GeneProduct] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.record.id

    @property
    def description(self) -> str:
        return self.record.description

    @property
    def nt_seq(self) -> str:
        return str(self.record.seq)

    @property
    def has_stop_codon(self) -> bool:
        """``True`` if the translated sequence contains a stop codon (``*``)."""
        return self.aa_seq is not None and "*" in self.aa_seq

    @property
    def has_ambiguous(self) -> bool:
        """``True`` if the translated sequence contains an ambiguous residue
        (``X``), typically caused by ambiguous nucleotides in the source."""
        return self.aa_seq is not None and "X" in self.aa_seq

    def get_product(self, name: str) -> GeneProduct | None:
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
    host: str | None = None
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

    def get_all_products(self) -> list[tuple[str | None, GeneProduct]]:
        """Return ``(segment_name, product)`` pairs for every alternative
        product across all sequences in the group."""
        results = []
        for seq in self.sequences:
            for p in seq.alt_products:
                results.append((seq.segment_name, p))
        return results
