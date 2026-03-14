"""Generation of influenza A alternative protein products.

Supports four mechanisms: direct translation, mRNA splicing, alternative
ORF scanning, and ribosomal frameshifting.
"""

from __future__ import annotations

from dataclasses import dataclass

from flua.constants import ALTERNATIVE_PRODUCTS
from flua.seq_utils import translate_frame1


@dataclass
class AlternativeProduct:
    """A single alternative protein product derived from an influenza
    segment."""

    name: str
    mechanism: str
    description: str
    nucleotide_seq: str
    aa_seq: str

    @property
    def has_stop_codon(self) -> bool:
        """``True`` if the translated sequence contains a stop codon (``*``)."""
        return "*" in self.aa_seq

    @property
    def has_ambiguous(self) -> bool:
        """``True`` if the translated sequence contains an ambiguous residue
        (``X``), typically caused by ambiguous nucleotides in the source."""
        return "X" in self.aa_seq


# ── Per-mechanism generators ─────────────────────────────────────────────


def _generate_direct(seq: str, pdef: dict) -> AlternativeProduct | None:
    """Primary protein: full-length frame-1 translation."""
    protein = translate_frame1(seq)
    return AlternativeProduct(
        name=pdef["name"],
        mechanism="direct",
        description=pdef["description"],
        nucleotide_seq=seq,
        aa_seq=protein,
    )


def _generate_spliced(seq: str, pdef: dict) -> AlternativeProduct | None:
    """Spliced product: join exon 1 + exon 2 then translate."""
    exon1_end = pdef["exon1_end"]
    exon2_start = pdef["exon2_start"]

    if len(seq) < exon2_start:
        return None

    spliced_nt = seq[:exon1_end] + seq[exon2_start:]
    protein = translate_frame1(spliced_nt)

    return AlternativeProduct(
        name=pdef["name"],
        mechanism="splicing",
        description=pdef["description"],
        nucleotide_seq=spliced_nt,
        aa_seq=protein,
    )


def _generate_alt_orf(seq: str, pdef: dict) -> AlternativeProduct | None:
    """Alternative ORF: scan *scan_frame* for the first ATG and translate
    to the first stop codon."""
    scan_frame = pdef.get("scan_frame", 1)
    min_length_aa = pdef.get("min_length_aa", 50)

    for i in range(scan_frame, len(seq) - 2, 3):
        codon = seq[i : i + 3].upper().replace("U", "T")
        if codon == "ATG":
            orf_seq = seq[i:]
            protein = translate_frame1(orf_seq)
            if "*" in protein:
                protein = protein[: protein.index("*")]
            if len(protein) >= min_length_aa:
                orf_nt = seq[i : i + (len(protein) + 1) * 3]
                return AlternativeProduct(
                    name=pdef["name"],
                    mechanism="alt_orf",
                    description=pdef["description"],
                    nucleotide_seq=orf_nt,
                    aa_seq=protein,
                )
    return None


def _generate_frameshift(seq: str, pdef: dict) -> AlternativeProduct | None:
    """Ribosomal frameshift: N-terminal domain (frame 0) fused with
    C-terminal domain (+*shift* frame)."""
    fs_nt = pdef["frameshift_nt"]
    shift = pdef.get("shift", 1)
    x_orf_aa_len = pdef.get("x_orf_length_aa", 61)

    if len(seq) < fs_nt + 10:
        return None

    n_term_protein = translate_frame1(seq[:fs_nt])

    c_term_start = fs_nt + shift
    c_term_full = translate_frame1(seq[c_term_start:])
    if "*" in c_term_full:
        c_term_full = c_term_full[: c_term_full.index("*")]
    c_term_protein = c_term_full[:x_orf_aa_len]

    fusion_protein = n_term_protein + c_term_protein
    fusion_nt = seq[: c_term_start + len(c_term_protein) * 3]

    return AlternativeProduct(
        name=pdef["name"],
        mechanism="frameshift",
        description=pdef["description"],
        nucleotide_seq=fusion_nt,
        aa_seq=fusion_protein,
    )


_GENERATORS = {
    "direct": _generate_direct,
    "splicing": _generate_spliced,
    "alt_orf": _generate_alt_orf,
    "frameshift": _generate_frameshift,
}


# ── Public API ───────────────────────────────────────────────────────────


def generate_alternative_products(
    sequence: str,
    segment_name: str,
    product_defs: dict[str, list[dict]] | None = None,
) -> list[AlternativeProduct]:
    """Generate all alternative products for a given segment sequence.

    Parameters
    ----------
    sequence:
        The nucleotide sequence of the segment.
    segment_name:
        One of the standard influenza A segment names (e.g. ``"PA"``).
    product_defs:
        Custom product definition table.  Defaults to
        :data:`~flua.constants.ALTERNATIVE_PRODUCTS`.
    """
    if product_defs is None:
        product_defs = ALTERNATIVE_PRODUCTS

    defs = product_defs.get(segment_name, [])
    if not defs:
        defs = [{"name": segment_name, "mechanism": "direct", "description": ""}]

    products = []
    for pdef in defs:
        generator = _GENERATORS.get(pdef["mechanism"])
        if generator is None:
            continue
        product = generator(sequence, pdef)
        if product is not None:
            products.append(product)
    return products
