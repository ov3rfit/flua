"""FASTA file I/O and DataFrame conversion."""

from __future__ import annotations

import io
import warnings
from collections import Counter
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

from flua.constants import INFLUENZA_SEGMENTS
from flua.models import AnalyzedSequence, SequenceGroup
from flua.products import generate_alternative_products
from flua.seq_utils import (
    detect_sequence_type,
    extract_subtype,
    identify_segment,
    translate_sequence,
)

# ── Internal helpers ──────────────────────────────────────────────────────


def _build_sequence_group(
    records: Iterable[SeqRecord],
    group_name: str,
    source_file: str,
    segment_names: list[str] | None = None,
) -> SequenceGroup:
    """Convert an iterable of :class:`~Bio.SeqRecord.SeqRecord` objects into
    a :class:`SequenceGroup`.

    This is the shared core used by :func:`load_fasta` and
    :func:`load_fasta_string`.
    """
    group = SequenceGroup(group_name=group_name, source_file=source_file)
    detected_subtypes: list[str] = []

    for record in records:
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

        group.sequences.append(
            AnalyzedSequence(
                record=record,
                seq_type=seq_type,
                aa_seq=aa_seq,
                segment_name=segment,
                alt_products=alt_products,
            )
        )

    if detected_subtypes:
        group.subtype = Counter(detected_subtypes).most_common(1)[0][0]

    return group


def _build_gisaid_groups(
    records: Iterable[SeqRecord],
    source_file: str,
    segment_names: list[str],
) -> list[SequenceGroup]:
    """Convert an iterable of :class:`~Bio.SeqRecord.SeqRecord` objects from
    a GISAID EpiFlu file into a list of :class:`SequenceGroup` objects.

    This is the shared core used by :func:`load_gisaid_fasta` and
    :func:`load_gisaid_fasta_string`.
    """
    groups: dict[str, SequenceGroup] = {}

    for record in records:
        parsed = _parse_gisaid_header(record.id)

        strain = parsed["strain"] or record.id
        segment = parsed["segment"] or identify_segment(
            record.id, record.description, segment_names
        )
        raw_subtype = parsed["subtype"] or ""
        subtype = extract_subtype(record.id, raw_subtype + " " + record.description)

        seq_str = str(record.seq)
        seq_type = detect_sequence_type(seq_str)
        aa_seq = translate_sequence(seq_str, seq_type)

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

        if strain not in groups:
            groups[strain] = SequenceGroup(
                group_name=strain,
                source_file=source_file,
                subtype=subtype,
            )
        elif subtype and not groups[strain].subtype:
            groups[strain].subtype = subtype

        groups[strain].sequences.append(analyzed)

    return list(groups.values())


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
    records = SeqIO.parse(str(filepath), "fasta")
    return _build_sequence_group(records, group_name, filepath.name, segment_names)


def load_fasta_string(
    fasta_text: str,
    group_name: str,
    segment_names: list[str] | None = None,
) -> SequenceGroup:
    """Parse a FASTA-formatted string and return a :class:`SequenceGroup`.

    Useful when the caller already holds the file contents in memory (e.g.
    a web backend that received an uploaded file).

    Parameters
    ----------
    fasta_text:
        Raw FASTA text (one or more records).
    group_name:
        Name assigned to the resulting group.  Unlike :func:`load_fasta`
        there is no filename to derive this from, so it is required.
    segment_names:
        Segment names to recognise.  Defaults to
        :data:`~flua.constants.INFLUENZA_SEGMENTS`.
    """
    records = SeqIO.parse(io.StringIO(fasta_text), "fasta")
    return _build_sequence_group(records, group_name, "<string>", segment_names)


def _parse_gisaid_header(record_id: str) -> dict[str, str | None]:
    """Parse a pipe-delimited GISAID EpiFlu FASTA header.

    Expected format: ``accession|strain|seg_num|segment|subtype|...``

    Returns a dict with ``accession``, ``strain``, ``segment``, and
    ``subtype`` keys.  Missing fields are ``None``.
    """
    parts = record_id.split("|")
    return {
        "accession": parts[0] if len(parts) > 0 else None,
        "strain": parts[1] if len(parts) > 1 else None,
        "segment": parts[3] if len(parts) > 3 else None,
        "subtype": parts[4] if len(parts) > 4 else None,
    }


def load_gisaid_fasta(
    filepath: str | Path,
    segment_names: list[str] | None = None,
) -> list[SequenceGroup]:
    """Read a multi-strain GISAID EpiFlu FASTA file.

    GISAID bulk downloads contain sequences from many strains in a single
    file with pipe-delimited headers::

        >EPI_ISL_123456|A/swine/Germany/R314/2015|4|HA|H1N1|...

    Each unique strain becomes one :class:`SequenceGroup`, and each
    segment record within that strain becomes an
    :class:`AnalyzedSequence`.

    Parameters
    ----------
    filepath:
        Path to the GISAID FASTA file.
    segment_names:
        Segment names to recognise.  Defaults to
        :data:`~flua.constants.INFLUENZA_SEGMENTS`.

    Returns
    -------
    list[SequenceGroup]
        One group per strain, in the order first encountered.
    """
    filepath = Path(filepath)
    if segment_names is None:
        segment_names = INFLUENZA_SEGMENTS
    records = SeqIO.parse(str(filepath), "fasta")
    return _build_gisaid_groups(records, filepath.name, segment_names)


def load_gisaid_fasta_string(
    fasta_text: str,
    segment_names: list[str] | None = None,
) -> list[SequenceGroup]:
    """Parse a GISAID EpiFlu FASTA-formatted string.

    The string equivalent of :func:`load_gisaid_fasta`, useful when the
    caller already holds the file contents in memory.

    Parameters
    ----------
    fasta_text:
        Raw FASTA text in GISAID EpiFlu pipe-delimited header format.
    segment_names:
        Segment names to recognise.  Defaults to
        :data:`~flua.constants.INFLUENZA_SEGMENTS`.

    Returns
    -------
    list[SequenceGroup]
        One group per strain, in the order first encountered.
    """
    if segment_names is None:
        segment_names = INFLUENZA_SEGMENTS
    records = SeqIO.parse(io.StringIO(fasta_text), "fasta")
    return _build_gisaid_groups(records, "<string>", segment_names)


def load_multiple_fasta(
    filepaths: list[str | Path],
    segment_names: list[str] | None = None,
) -> list[SequenceGroup]:
    """Read multiple FASTA files and return a list of
    :class:`SequenceGroup` objects."""
    return [load_fasta(fp, segment_names=segment_names) for fp in filepaths]


# ── DataFrame conversion ─────────────────────────────────────────────────


def _is_flagged(
    seq_obj: AnalyzedSequence,
    exclude_stop_codons: bool,
    exclude_ambiguous: bool,
) -> bool:
    """Return ``True`` if *seq_obj* should be excluded from the DataFrame
    based on the caller's quality filters."""
    if exclude_stop_codons and seq_obj.has_stop_codon:
        return True
    if exclude_ambiguous and seq_obj.has_ambiguous:
        return True
    return False


def _is_product_flagged(
    product: AlternativeProduct,
    exclude_stop_codons: bool,
    exclude_ambiguous: bool,
) -> bool:
    """Return ``True`` if *product* should be excluded from the DataFrame."""
    if exclude_stop_codons and product.has_stop_codon:
        return True
    if exclude_ambiguous and product.has_ambiguous:
        return True
    return False


def groups_to_dataframe(
    groups: list[SequenceGroup],
    value_type: Literal["raw", "translated"] = "raw",
    segment_names: list[str] | None = None,
    include_alt_products: bool = True,
    exclude_stop_codons: bool = False,
    exclude_ambiguous: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
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
    exclude_stop_codons:
        If ``True``, set the sequence value to ``None`` for any segment
        or alternative product whose translated sequence contains a stop
        codon (``*``).  Applies to both ``"raw"`` and ``"translated"``
        outputs.
    exclude_ambiguous:
        If ``True``, set the sequence value to ``None`` for any segment
        or alternative product whose translated sequence contains an
        ambiguous residue (``X``).

    Returns
    -------
    tuple[pandas.DataFrame, list[str]]
        A ``(df, warnings)`` tuple where ``warnings`` is a list of
        human-readable messages about sequence-length inconsistencies
        across samples.  An empty list means no issues were detected.
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

            if seq_obj is None or _is_flagged(
                seq_obj, exclude_stop_codons, exclude_ambiguous
            ):
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
                        if product is None or _is_product_flagged(
                            product, exclude_stop_codons, exclude_ambiguous
                        ):
                            row[col_key] = None
                        else:
                            row[col_key] = product.aa_seq
                    else:
                        row[col_key] = None

        rows.append(row)

    df = pd.DataFrame(rows)
    warn_messages = _check_seq_length_consistency(df, segment_names, seq_suffix)
    return df, warn_messages


def _check_seq_length_consistency(
    df: pd.DataFrame,
    segment_names: list[str],
    seq_suffix: str,
) -> list[str]:
    """Return warning messages when sequence lengths for the same segment
    differ across samples."""
    messages: list[str] = []
    if len(df) < 2:
        return messages
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
            messages.append(
                f"[{seg_name}] Sequence lengths differ across samples: {info}"
            )
    return messages
