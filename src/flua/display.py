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
        trans_len = len(seq.translated) if seq.translated else "-"
        print(
            f"  [{seg_label}] {seq.id} | "
            f"type={seq.seq_type} | "
            f"length={seq.length} | "
            f"translated_length={trans_len}"
        )
        for p in seq.alt_products:
            print(
                f"    └─ {p.name} ({p.mechanism}): {p.length_aa} aa | {p.description}"
            )
    print()
