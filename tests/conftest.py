"""Shared fixtures for the flua test suite."""

from __future__ import annotations

import random
from pathlib import Path

import pytest


def _make_coding_seq(length: int, seed: int = 42) -> str:
    """Generate a deterministic pseudo-random coding sequence."""
    rng = random.Random(seed)
    return "ATG" + "".join(rng.choice("ATGC") for _ in range(length - 3))


@pytest.fixture()
def h1n1_fasta(tmp_path: Path) -> Path:
    """Write a complete 8-segment H1N1 FASTA file and return its path."""
    segments = [
        ("PB2", 2340),
        ("PB1", 2340),
        ("PA", 2232),
        ("HA", 1776),
        ("NP", 1566),
        ("NA", 1413),
        ("MP", 1027),
        ("NS", 890),
    ]
    lines: list[str] = []
    for i, (seg, length) in enumerate(segments):
        lines.append(f">seg{i + 1}|{seg}|A/California/07/2009(H1N1)")
        lines.append(_make_coding_seq(length, seed=42 + i))
    fp = tmp_path / "test_h1n1.fasta"
    fp.write_text("\n".join(lines) + "\n")
    return fp


@pytest.fixture()
def h5n1_fasta(tmp_path: Path) -> Path:
    """Write a 7-segment H5N1 FASTA file (missing NS) and return its
    path."""
    segments = [
        ("PB2", 2340),
        ("PB1", 2340),
        ("PA", 2232),
        ("HA", 1776),
        ("NP", 1566),
        ("NA", 1413),
        ("MP", 1027),
    ]
    lines: list[str] = []
    for i, (seg, length) in enumerate(segments):
        lines.append(f">seg{i + 1}|{seg}|A/Vietnam/1203/2004(H5N1)")
        lines.append(_make_coding_seq(length, seed=100 + i))
    fp = tmp_path / "test_h5n1.fasta"
    fp.write_text("\n".join(lines) + "\n")
    return fp


@pytest.fixture()
def unknown_fasta(tmp_path: Path) -> Path:
    """Write a 2-segment FASTA file with no subtype info."""
    lines = [
        ">seg1|PB2|unknown_virus_sample",
        _make_coding_seq(2340, seed=200),
        ">seg2|PB1|unknown_virus_sample",
        _make_coding_seq(2340, seed=201),
    ]
    fp = tmp_path / "test_unknown.fasta"
    fp.write_text("\n".join(lines) + "\n")
    return fp
