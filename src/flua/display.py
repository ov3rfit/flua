"""Human-readable display helpers."""

from __future__ import annotations

from flua.models import SequenceGroup


def print_group_summary(group: SequenceGroup) -> None:
    """Print a concise summary of a :class:`SequenceGroup` to stdout."""
    print(f"=== {group.group_name} (from: {group.source_file}) ===")
    print(f"  Subtype: {group.subtype or '(not detected)'}")
    print(f"  Total sequences: {len(group.sequences)}")
    for seq in group.sequences:
        seg_label = seq.segment_name or "(unknown)"
        aa_len = len(seq.aa_seq) if seq.aa_seq else "-"
        print(
            f"  [{seg_label}] {seq.id} | "
            f"type={seq.seq_type} | "
            f"length={len(seq.nt_seq)} | "
            f"aa_length={aa_len}"
        )
        for p in seq.alt_products:
            print(
                f"    └─ {p.name} ({p.mechanism}): {len(p.aa_seq)} aa"
            )
    print()
