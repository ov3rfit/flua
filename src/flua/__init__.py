"""flua – Influenza A FASTA sequence analysis toolkit."""

from flua.constants import ALTERNATIVE_PRODUCTS, INFLUENZA_SEGMENTS
from flua.display import print_group_summary
from flua.io import (
    groups_to_dataframe,
    load_fasta,
    load_fasta_string,
    load_gisaid_fasta,
    load_gisaid_fasta_string,
    load_multiple_fasta,
)
from flua.models import AnalyzedSequence, SequenceGroup
from flua.products import AlternativeProduct, generate_alternative_products
from flua.seq_utils import (
    detect_sequence_type,
    extract_subtype,
    identify_segment,
    translate_sequence,
)

__all__ = [
    # Constants
    "ALTERNATIVE_PRODUCTS",
    "INFLUENZA_SEGMENTS",
    # Models
    "AlternativeProduct",
    "AnalyzedSequence",
    "SequenceGroup",
    # Functions
    "detect_sequence_type",
    "extract_subtype",
    "generate_alternative_products",
    "groups_to_dataframe",
    "identify_segment",
    "load_fasta",
    "load_fasta_string",
    "load_gisaid_fasta",
    "load_gisaid_fasta_string",
    "load_multiple_fasta",
    "print_group_summary",
    "translate_sequence",
]
