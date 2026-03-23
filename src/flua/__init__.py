"""flua – Influenza A FASTA sequence analysis toolkit."""

from flua.constants import (
    AA_ALPHABET,
    AA_ALPHABET_EXTENDED,
    GENE_PRODUCTS,
    INFLUENZA_SEGMENTS,
    NT_ALPHABET,
)
from flua.display import print_group_summary
from flua.encoding import (
    PositionalOneHotEncoder,
    sequences_to_label_encoding,
    sequences_to_ohe_tensor,
)
from flua.io import (
    groups_to_dataframe,
    load_fasta,
    load_fasta_string,
    load_gisaid_fasta,
    load_gisaid_fasta_string,
    load_multiple_fasta,
)
from flua.ml import (
    check_length_consistency,
    encode_subtype,
    sequences_to_composition,
    sequences_to_kmer_freq,
)
from flua.models import AnalyzedSequence, SequenceGroup
from flua.products import GeneProduct, generate_gene_products
from flua.seq_utils import (
    detect_sequence_type,
    extract_subtype,
    identify_segment,
    translate_sequence,
)

__all__ = [
    # Constants
    "AA_ALPHABET",
    "AA_ALPHABET_EXTENDED",
    "NT_ALPHABET",
    "GENE_PRODUCTS",
    "INFLUENZA_SEGMENTS",
    # Models
    "GeneProduct",
    "AnalyzedSequence",
    "SequenceGroup",
    # Functions
    "detect_sequence_type",
    "extract_subtype",
    "generate_gene_products",
    "groups_to_dataframe",
    "identify_segment",
    "load_fasta",
    "load_fasta_string",
    "load_gisaid_fasta",
    "load_gisaid_fasta_string",
    "load_multiple_fasta",
    "print_group_summary",
    "translate_sequence",
    # Positional encoders
    "PositionalOneHotEncoder",
    "sequences_to_label_encoding",
    "sequences_to_ohe_tensor",
    # ML utilities
    "check_length_consistency",
    "encode_subtype",
    "sequences_to_composition",
    "sequences_to_kmer_freq",
]
