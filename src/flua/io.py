"""FASTA file I/O and DataFrame conversion."""

from __future__ import annotations

import warnings
from collections import Counter
from pathlib import Path
from typing import Literal

import pandas as pd
from Bio import SeqIO

from flua.constants import INFLUENZA_SEGMENTS
from flua.models import AnalyzedSequence, SequenceGroup
from flua.products import generate_alternative_products
from flua.seq_utils import (
    detect_sequence_type,
    extract_subtype,
    identify_segment,
    translate_sequence,
)

# ── Loading ──────────────────────────────────────────────────────────────


def load_fasta(
    filepath: str | Path,
    segment_names: list[str] | None = None,
    group_name: str | None = None,
) -> SequenceGroup:
    """Read a single FASTA file and return a :class:`SequenceGroup`."""
    filepath = Path(filepath)
    if group_name is None:
        group_name = filepath.stem

    group = SequenceGroup(group_name=group_name, source_file=str(filepath))
    detected_subtypes: list[str] = []

    for record in SeqIO.parse(str(filepath), "fasta"):
        seq_str = str(record.seq)
        seq_type = detect_sequence_type(seq_str)
        aa_seq = translate_sequence(seq_str, seq_type)
        segment = identify_segment(record.id, record.description, segment_names)

        subtype = extract_subtype(record.id, record.description)
        if subtype:
            detected_subtypes.append(subtype)

        alt_products: list = []
        if seq_type != "Protein" and segment is not None:
            alt_products = generate_alternative_products(seq_str, segment)

        analyzed = AnalyzedSequence(
            record=record,
            seq_type=seq_type,
            aa_seq=aa_seq,
            segment_name=segment,
            alt_products=alt_products,
        )
        group.sequences.append(analyzed)

    # Assign the most frequently detected subtype to the group.
    if detected_subtypes:
        group.subtype = Counter(detected_subtypes).most_common(1)[0][0]

    return group


def load_multiple_fasta(
    filepaths: list[str | Path],
    segment_names: list[str] | None = None,
) -> list[SequenceGroup]:
    """Read multiple FASTA files and return a list of
    :class:`SequenceGroup` objects."""
    return [load_fasta(fp, segment_names=segment_names) for fp in filepaths]


# ── DataFrame conversion ─────────────────────────────────────────────────


def groups_to_dataframe(
    groups: list[SequenceGroup],
    value_type: Literal["raw", "translated"] = "raw",
    segment_names: list[str] | None = None,
    include_alt_products: bool = True,
) -> pd.DataFrame:
    """Convert a list of :class:`SequenceGroup` objects into a
    :class:`~pandas.DataFrame`.

    Parameters
    ----------
    groups:
        Sequence groups to convert.
    value_type:
        ``"raw"`` for nucleotide sequences, ``"translated"`` for amino
        acid sequences.
    segment_names:
        Segment names to include as columns.  Defaults to
        :data:`~flua.constants.INFLUENZA_SEGMENTS`.
    include_alt_products:
        If ``True``, add columns for non-direct alternative products
        (e.g. spliced, frameshift, alt_orf).
    """
    if segment_names is None:
        segment_names = INFLUENZA_SEGMENTS

    seq_suffix = "_nt" if value_type == "raw" else "_aa"

    # Collect non-direct alternative product names across all groups.
    all_alt_product_names: dict[str, set[str]] = {}
    if include_alt_products:
        for group in groups:
            for seq in group.sequences:
                if seq.segment_name:
                    for p in seq.alt_products:
                        if p.mechanism != "direct":
                            if seq.segment_name not in all_alt_product_names:
                                all_alt_product_names[seq.segment_name] = set()
                            all_alt_product_names[seq.segment_name].add(p.name)

    rows = []
    for group in groups:
        row: dict = {
            "group_name": group.group_name,
            "source_file": group.source_file,
            "subtype": group.subtype,
            "num_sequences": len(group.sequences),
        }

        for seg_name in segment_names:
            seq_obj = group.get_segment(seg_name)

            if seq_obj is None:
                row[f"{seg_name}{seq_suffix}"] = None
            else:
                if value_type == "translated" and seq_obj.aa_seq is not None:
                    row[f"{seg_name}{seq_suffix}"] = seq_obj.aa_seq
                else:
                    row[f"{seg_name}{seq_suffix}"] = seq_obj.nucleotide_seq

            if include_alt_products and seg_name in all_alt_product_names:
                for prod_name in sorted(all_alt_product_names[seg_name]):
                    col_key = f"{prod_name}_aa"
                    if seq_obj is not None:
                        product = seq_obj.get_product(prod_name)
                        row[col_key] = (
                            product.aa_seq if product is not None else None
                        )
                    else:
                        row[col_key] = None

        rows.append(row)

    df = pd.DataFrame(rows)
    _check_seq_length_consistency(df, segment_names, seq_suffix)
    return df


def _check_seq_length_consistency(
    df: pd.DataFrame,
    segment_names: list[str],
    seq_suffix: str,
) -> None:
    """Emit a warning when sequence lengths for the same segment differ
    across samples."""
    if len(df) < 2:
        return
    for seg_name in segment_names:
        seq_col = f"{seg_name}{seq_suffix}"
        if seq_col not in df.columns:
            continue
        lengths = df[seq_col].dropna().str.len()
        if len(lengths) < 2:
            continue
        if len(lengths.unique()) > 1:
            info = ", ".join(
                f"{r['group_name']}={len(r[seq_col])}"
                for _, r in df.iterrows()
                if pd.notna(r[seq_col])
            )
            warnings.warn(
                f"[{seg_name}] Sequence lengths differ across samples: {info}",
                UserWarning,
                stacklevel=3,
            )
