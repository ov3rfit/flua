"""Constants and configuration for influenza A sequence analysis."""

from __future__ import annotations

# Standard influenza A segment names (8 segments).
INFLUENZA_SEGMENTS = ["PB2", "PB1", "PA", "HA", "NP", "NA", "MP", "NS"]

# Additional tokens that should resolve to a canonical segment name when
# scanning FASTA headers.  The canonical name (dict key) is what gets
# returned; every token in the value list is accepted as a synonym.  The
# canonical token itself does not need to be repeated here — callers add it
# automatically.
#
# ``MP`` ↔ ``M``: GenBank often labels segment 7 as ``M`` while NCBI
# Influenza Virus Resource and GISAID use ``MP``.  Accept both, return ``MP``.
SEGMENT_ALIASES: dict[str, list[str]] = {
    "MP": ["M"],
}

# IUPAC amino acid characters that never appear in DNA/RNA sequences.
PROTEIN_ONLY_CHARS = set("FLIMSPHQEDKWRV")

# ---------------------------------------------------------------------------
# Sequence alphabets
# ---------------------------------------------------------------------------

#: Standard nucleotide alphabet (DNA).
NT_ALPHABET: str = "ACGT"

#: Standard 20 amino-acid alphabet (IUPAC single-letter codes, uppercase).
AA_ALPHABET: str = "ACDEFGHIKLMNPQRSTVWY"

#: Extended amino-acid alphabet: standard 20 + stop codon (``*``) + ambiguous
#: residue (``X``).  Use this when real-world sequences may contain stop
#: codons (frameshifts, sequencing artefacts) or IUPAC ambiguity codes.
AA_ALPHABET_EXTENDED: str = "ACDEFGHIKLMNPQRSTVWY*X"

# ---------------------------------------------------------------------------
# Influenza A alternative product definitions
# ---------------------------------------------------------------------------
# Mechanism types:
#   "direct"     – translate the full-length sequence in frame 1
#   "splicing"   – join exon1 + exon2, then translate
#   "alt_orf"    – scan a specified reading frame for the first ATG
#   "frameshift" – ribosomal frameshift: N-terminal frame 0, then +1 frame
#
# Coordinates are 0-based and correspond to a canonical influenza A genome.
# Exact positions may vary slightly between strains.
# ---------------------------------------------------------------------------

GENE_PRODUCTS: dict[str, list[dict]] = {
    "PB2": [
        {"name": "PB2", "mechanism": "direct"},
    ],
    "PB1": [
        {"name": "PB1", "mechanism": "direct"},
        {
            "name": "PB1-F2",
            "mechanism": "alt_orf",
            "scan_frame": 1,
            "min_length_aa": 10,
        },
    ],
    "PA": [
        {"name": "PA", "mechanism": "direct"},
        {
            "name": "PA-X",
            "mechanism": "frameshift",
            "frameshift_nt": 573,
            "shift": 1,
            "x_orf_length_aa": 61,
        },
    ],
    "HA": [
        {"name": "HA", "mechanism": "direct"},
    ],
    "NP": [
        {"name": "NP", "mechanism": "direct"},
    ],
    "NA": [
        {"name": "NA", "mechanism": "direct"},
    ],
    "MP": [
        {"name": "M1", "mechanism": "direct"},
        {"name": "M2", "mechanism": "splicing", "exon1_end": 26, "exon2_start": 715},
        # {"name": "M2", "mechanism": "splicing", "exon1_end": 51, "exon2_start": 740},
    ],
    "NS": [
        {"name": "NS1", "mechanism": "direct"},
        {"name": "NEP", "mechanism": "splicing", "exon1_end": 56, "exon2_start": 529},
    ],
}
