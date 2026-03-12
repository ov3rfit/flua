"""Constants and configuration for influenza A sequence analysis."""

from __future__ import annotations

# Standard influenza A segment names (8 segments).
INFLUENZA_SEGMENTS = ["PB2", "PB1", "PA", "HA", "NP", "NA", "MP", "NS"]

# IUPAC amino acid characters that never appear in DNA/RNA sequences.
PROTEIN_ONLY_CHARS = set("FLIMSPHQEDKWRV")

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

ALTERNATIVE_PRODUCTS: dict[str, list[dict]] = {
    "PB2": [
        {
            "name": "PB2",
            "mechanism": "direct",
            "description": "RNA-dependent RNA polymerase subunit PB2",
        },
    ],
    "PB1": [
        {
            "name": "PB1",
            "mechanism": "direct",
            "description": "RNA-dependent RNA polymerase subunit PB1",
        },
        {
            "name": "PB1-F2",
            "mechanism": "alt_orf",
            "description": "Pro-apoptotic mitochondrial protein from +1 ORF of PB1",
            "scan_frame": 1,
            "min_length_aa": 50,
        },
    ],
    "PA": [
        {
            "name": "PA",
            "mechanism": "direct",
            "description": "RNA-dependent RNA polymerase subunit PA",
        },
        {
            "name": "PA-X",
            "mechanism": "frameshift",
            "description": "Host shutoff protein via +1 ribosomal frameshift of PA",
            "frameshift_nt": 573,  # 191 codons * 3
            "shift": 1,
            "x_orf_length_aa": 61,
        },
    ],
    "HA": [
        {
            "name": "HA",
            "mechanism": "direct",
            "description": "Hemagglutinin",
        },
    ],
    "NP": [
        {
            "name": "NP",
            "mechanism": "direct",
            "description": "Nucleoprotein",
        },
    ],
    "NA": [
        {
            "name": "NA",
            "mechanism": "direct",
            "description": "Neuraminidase",
        },
    ],
    "MP": [
        {
            "name": "M1",
            "mechanism": "direct",
            "description": "Matrix protein 1 (unspliced colinear transcript)",
        },
        {
            "name": "M2",
            "mechanism": "splicing",
            "description": "Ion channel protein (spliced from MP segment)",
            "exon1_end": 51,
            "exon2_start": 740,
        },
    ],
    "NS": [
        {
            "name": "NS1",
            "mechanism": "direct",
            "description": "Non-structural protein 1 (unspliced colinear transcript)",
        },
        {
            "name": "NEP",
            "mechanism": "splicing",
            "description": "Nuclear export protein / NS2 (spliced from NS segment)",
            "exon1_end": 56,
            "exon2_start": 529,
        },
    ],
}
