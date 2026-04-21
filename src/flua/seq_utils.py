"""Low-level sequence utilities: type detection, translation, subtype
extraction, and segment identification."""

from __future__ import annotations

import re
from typing import Literal

from Bio.Seq import Seq

from flua.constants import INFLUENZA_SEGMENTS, PROTEIN_ONLY_CHARS, SEGMENT_ALIASES

# ── Sequence type detection ──────────────────────────────────────────────


def detect_sequence_type(sequence: str) -> Literal["DNA", "RNA", "Protein"]:
    """Classify sequence as DNA, RNA, or Protein based on its character
    composition.

    If more than 90% of characters are standard nucleotides (G, C, A, T),
    the sequence is classified as DNA or RNA regardless of other characters.
    This prevents degenerate IUPAC nucleotide codes (e.g. S, R, K, M, V, H,
    W) from being confused with protein-only amino acid characters.
    """
    seq_upper = sequence.upper().replace("-", "").replace(".", "")
    if not seq_upper:
        return "DNA"
    unique_chars = set(seq_upper)

    gcat_ratio = sum(1 for c in seq_upper if c in "GCAT") / len(seq_upper)
    if gcat_ratio > 0.9:
        if "U" in unique_chars and "T" not in unique_chars:
            return "RNA"
        return "DNA"

    if unique_chars & PROTEIN_ONLY_CHARS:
        return "Protein"
    if "U" in unique_chars and "T" not in unique_chars:
        return "RNA"
    return "DNA"


# ── Translation ──────────────────────────────────────────────────────────


def translate_frame1(sequence: str) -> str:
    """Translate sequence in reading frame 1 (offset 0).

    Trailing nucleotides that do not form a complete codon are discarded.
    """
    seq_obj = Seq(sequence)
    trimmed = seq_obj[: len(seq_obj) - len(seq_obj) % 3]
    if len(trimmed) == 0:
        return ""
    return str(trimmed.translate())


def translate_sequence(sequence: str, seq_type: str) -> str | None:
    """Return the frame-1 translation of sequence, or ``None`` if it is
    already a protein."""
    if seq_type == "Protein":
        return None
    return translate_frame1(sequence)


# ── Subtype extraction ───────────────────────────────────────────────────

# H<digits>N<digits> with optional parentheses and pdm suffix.
_SUBTYPE_PATTERN = re.compile(
    r"[\(\[\s|_/]?"
    r"(H\d{1,2}N\d{1,2})"
    r"[\)\]\s|_/]?"
    r"(pdm\d{0,4})?"
    r"[\)\]\s|_/]?",
    re.IGNORECASE,
)

# Separate H and N tokens (e.g. "H5 subtype N6").
_H_PATTERN = re.compile(r"(?<![A-Z0-9])(H\d{1,2})(?![A-Z0-9])", re.IGNORECASE)
_N_PATTERN = re.compile(r"(?<![A-Z0-9])(N\d{1,2})(?![A-Z0-9])", re.IGNORECASE)


def extract_subtype(header_id: str, description: str) -> str | None:
    """Extract an influenza subtype string from a FASTA header.

    Supported formats include ``A/California/07/2009(H1N1)``,
    ``H5N1``, ``H1N1pdm09``, ``H3N2|segment 3``, and split
    H/N notation such as ``"H5 subtype N6"``.

    Returns
    -------
    str | None
        The extracted subtype (e.g. ``"H1N1"``, ``"H5N1pdm09"``), or
        ``None`` when no subtype is found.
    """
    combined = f"{header_id} {description}"

    match = _SUBTYPE_PATTERN.search(combined)
    if match:
        subtype = match.group(1).upper()
        pdm_suffix = match.group(2)
        if pdm_suffix:
            subtype += pdm_suffix.lower()
        return subtype

    h_match = _H_PATTERN.search(combined)
    n_match = _N_PATTERN.search(combined)
    if h_match and n_match:
        return h_match.group(1).upper() + n_match.group(1).upper()

    return None


# ── Segment identification ───────────────────────────────────────────────


def _build_segment_patterns(
    segment_names: list[str],
) -> list[tuple[str, re.Pattern]]:
    """Build compiled regex patterns for each segment name.

    Each canonical segment name may have alias tokens listed in
    :data:`~flua.constants.SEGMENT_ALIASES` — every token is accepted as a
    match but the canonical name is what gets returned (e.g. header
    ``"sample_M_flu"`` → segment ``"MP"``).

    Longer tokens are matched first so that e.g. ``"MP"`` takes priority
    over its alias ``"M"``, and ``"PB1-F2"`` over ``"PB1"``.  Word
    boundaries are defined by characters that are not alphanumeric or
    hyphens, preventing ``"PA"`` from matching inside ``"PA-X"`` or
    ``"M"`` from matching inside ``"M1"``.
    """
    name_token_pairs: list[tuple[str, str]] = []
    for canonical in segment_names:
        tokens = [canonical, *SEGMENT_ALIASES.get(canonical, [])]
        for tok in tokens:
            name_token_pairs.append((canonical, tok))

    name_token_pairs.sort(key=lambda pair: len(pair[1]), reverse=True)

    patterns = []
    for canonical, tok in name_token_pairs:
        escaped = re.escape(tok.upper())
        pattern = re.compile(
            r"(?<![A-Z0-9\-])" + escaped + r"(?![A-Z0-9\-])",
            re.IGNORECASE,
        )
        patterns.append((canonical, pattern))
    return patterns


def identify_segment(
    header_id: str,
    description: str,
    segment_names: list[str] | None = None,
) -> str | None:
    """Identify the influenza segment name from a FASTA header."""
    if segment_names is None:
        segment_names = INFLUENZA_SEGMENTS

    combined = f"{header_id} {description}"
    patterns = _build_segment_patterns(segment_names)

    for name, pattern in patterns:
        if pattern.search(combined):
            return name
    return None
