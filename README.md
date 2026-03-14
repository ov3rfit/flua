# flua

> **Note:** This project is under active development. APIs may change without notice.

Influenza A sequence analysis toolkit.

## Features

- Load and parse influenza A FASTA files (file path or string input)
- GISAID multi-segment FASTA support
- Automatic sequence type detection (DNA / RNA / Protein)
- Subtype extraction from FASTA headers (e.g. H1N1, H5N1pdm09)
- Segment identification (PB2, PB1, PA, HA, NP, NA, MP, NS)
- Translation with alternative product generation (splicing, frameshift, alt-ORF)
- DataFrame export for multi-sample comparative analysis

## Installation

```bash
pip install flua
```

## Usage

```python
from flua import load_fasta, load_multiple_fasta, load_gisaid_fasta, groups_to_dataframe

# Load a single FASTA file
group = load_fasta("sample.fasta")
print(group.subtype)  # e.g. "H1N1"

# Load from a FASTA-formatted string
group = load_fasta_string(fasta_text)

# Load a GISAID multi-segment FASTA
group = load_gisaid_fasta("gisaid_sample.fasta")

# Load multiple files and export to DataFrame
groups = load_multiple_fasta(["sample1.fasta", "sample2.fasta"])
df = groups_to_dataframe(groups, value_type="translated")
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src/ tests/
```
