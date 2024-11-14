import shlex
import numpy as np

from subprocess import PIPE, Popen

VOCAB = {"A": 1, "C": 2, "G": 3, "T": 4, "-": 5, "N": 6}
VOCAB_KMER = {"N": 0, "A": 1, "C": 2, "G": 3, "T": 4, "-": 5}
VOCAB_KMER_SIMULATE = {"N": 0, "A": 1, "C": 2, "G": 3, "T": 4}
REVERSE_VOCAB_KMER_SIMULATE = {
    v: k for k, v in VOCAB_KMER_SIMULATE.items()
}  # Reverse the VOCAB mapping
CLASSES_PROB_1 = ["-", "A", "C", "G", "T"]
CLASSES_PROB_2 = [
    "-",
    "A",
    "C",
    "G",
    "T",
    "AA",
    "AC",
    "AG",
    "AT",
    "CA",
    "CC",
    "CG",
    "CT",
    "GA",
    "GC",
    "GG",
    "GT",
    "TA",
    "TC",
    "TG",
    "TT",
    "rep3",
    "rep4",
    "rep5",
    "rep6",
]
TOKENS = ["SEP"] + CLASSES_PROB_2
KMER_TOKEN_SHIFT = 30  # the start value of kmer encoded tokens


def reverse_complement(sequence):
    """
    Reverse complement the input nucleotide sequence.

    Args:
    - sequence (str): Input nucleotide sequence.

    Returns:
    - str: Reverse complemented nucleotide sequence.
    """
    # Reverse the sequence
    reversed_sequence = sequence[::-1]

    # Map each nucleotide to its complement
    complement_mapping = {"A": "T", "T": "A", "C": "G", "G": "C", "-": "-"}

    # Complement each nucleotide
    complemented_sequence = "".join(
        complement_mapping.get(nuc, nuc) for nuc in reversed_sequence
    )

    return complemented_sequence


def one_hot_encoding(array, num_class: int):
    """
    只对 A,C,G,T,-,[PAD] encoding
    A: [1, 0, 0, 0, 0], ..., [PAD]: [0, 0, 0, 0, 0]
    """
    output = []
    mapping = np.eye(num_class)
    for value in array:  # type: ignore
        if value == 0:
            output.append(np.zeros(num_class))
        else:
            output.append(np.eye(num_class)[int(value) - 1])
    output = np.array(output, dtype=np.float32)
    return output


def print_label(
    chrom: str,
    type: str,
    read_strand: str,
    position: int,
    label: str,
    read_base: str,
    ref_base: str,
    alts: str,
    is_germline: str,
    variant_type: str,
    sequence_around: str,
    file_p: str,
):
    """
    Print label to output config.data_path_label_f file.
    position should be change to 1-base
    """
    label_f = open(file_p, "a")
    print(
        f"chrom:{chrom} "
        f"type:{type} "
        f"read_strand:{read_strand} "
        f"pos_ref:{position} "
        f"label:{label} "
        f"read_base:{read_base} "
        f"ref_base:{ref_base} "
        f"alts:{alts} "
        f"is_germline:{is_germline} "
        f"variant_type:{variant_type} "
        f"sequence_around:{sequence_around} ",
        flush=True,
        file=label_f,
    )


def samtools_view_from(ctg_name, ctg_start, ctg_end, bam_fn, min_mq, samtools):
    """
    Get reads record by 'samtools view'
    """
    region = ""
    if ctg_name:
        region = "%s:%d-%d" % (ctg_name, ctg_start, ctg_end)

    subprocess = Popen(
        shlex.split(f"{samtools} view -q {min_mq} {bam_fn} {region}"), stdout=PIPE
    )
    return subprocess


def samtools_faidx_from(ctg_name, ctg_start, ctg_end, fasta_fn, samtools):
    region = "%s:%d-%d" % (ctg_name, ctg_start, ctg_end)
    subprocess = Popen(
        shlex.split(f"{samtools} faidx {fasta_fn} {region}"), stdout=PIPE
    )
    return subprocess


def decode_kmer(kmer_encodings, k=3):
    len_vocab = len(VOCAB_KMER_SIMULATE)
    decoded_seq = []

    for kmer_value in kmer_encodings:
        decoded_kmer = []
        value = int(kmer_value)

        for i in range(k):
            char_index = value // len_vocab ** (k - i - 1)
            decoded_kmer.append(REVERSE_VOCAB_KMER_SIMULATE[char_index])
            value %= len_vocab ** (k - i - 1)

        decoded_seq.append("".join(decoded_kmer))

    return "".join([kmer[0] for kmer in decoded_seq[:-1]]) + decoded_seq[-1]


def input_tokenization_kmer(input_seq: str, k=3):
    """
    Encode input sequence with k-mer
    """
    len_seq = len(input_seq)
    len_vocab = len(VOCAB_KMER)  # for real case
    # len_vocab = len(utils.VOCAB_KMER_SIMULATE)  # for simulate data

    # Initialize an empty list to store k-mer encoding
    kmer_encodings = []

    # Check if the sequence length is at least the k-mer length
    if len_seq < k:
        return None

    # Iterate through the sequence to extract k-mers
    for i in range(len_seq - k + 1):
        kmer = input_seq[i : i + k]

        # Convert k-mer into numerical encoding based on VOCAB
        kmer_value = 0
        for j, char in enumerate(kmer):
            if char not in VOCAB_KMER:  # for real data
                # if char not in utils.VOCAB_KMER_SIMULATE:  # for simulate data
                return None  # Handle unkwon characters

            # Calculate the unique value for the k-mer
            # kmer_value += utils.VOCAB_KMER_SIMULATE[char] * (
            #    len_vocab ** (k - j - 1)
            # )  # for simulate data
            kmer_value += VOCAB_KMER[char] * (len_vocab ** (k - j - 1))  # for real data

        # Append the encoded k-mer value to the list
        # kmer_encodings.append(kmer_value)
        kmer_encodings.append(kmer_value + KMER_TOKEN_SHIFT)

    # Convert the list to a numpy array for efficient processing
    # kmer_encodings = np.array(kmer_encodings, dtype=np.float32)
    return kmer_encodings


def input_tokenization_include_ground_truth_kmer(
    up_seq: str, next_base: str, insertion: str, kmer: int
):
    input_array = []
    input_array = input_tokenization_kmer(up_seq, kmer)

    if input_array is None:
        return None
    input_array.append(TOKENS.index("SEP"))
    input_array.append(TOKENS.index(next_base))
    input_array.append(TOKENS.index(insertion))
    return input_array

def input_tokenization_without_grount_truth_kmer(
    up_seq:str, kmer:int
):
    input_array = []
    input_array = input_tokenization_kmer(up_seq, kmer)
    
    if input_array is None:
        return None
    return input_array
