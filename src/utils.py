import numpy as np

VOCAB = {"A": 1, "C": 2, "G": 3, "T": 4, "-": 5, "N": 6}
VOCAB_KMER = {"N": 0, "A": 1, "C": 2, "G": 3, "T": 4, "-": 5}
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

SNVS = [
    ("A", "A"),
    ("A", "C"),
    ("A", "G"),
    ("A", "T"),
    ("A", "-"),
    ("C", "C"),
    ("C", "G"),
    ("C", "T"),
    ("C", "-"),
    ("G", "G"),
    ("G", "T"),
    ("G", "-"),
    ("T", "T"),
    ("T", "-"),
]

GENOTYPES = ["0/0", "1/1", "0/1", "1/2"]

IUPAC_base_to_ACGT_base_dict = dict(
    zip(
        "ACGTURYSWKMBDHVNP-",
        (
            "A",
            "C",
            "G",
            "T",
            "T",
            "A",
            "C",
            "C",
            "A",
            "G",
            "A",
            "C",
            "A",
            "A",
            "A",
            "N",
            "P",
            "-",
        ),
    )
)


class ALLELES:
    """
    example:
    snv  {'snv_A_C': ('A', 'C')}, the firtst allele is A, the second allele is C, is a heterozygous snv
    insertion:  'insertion_C_rep6': ('C', 'CC'), ...},
                the first allele is insert C, the second allele is insert CC, is a heterozygous insertion

    for insertion, "insertion_N_N" means no insertion, "insertion_A_A" means all insertions are A here.
                   "insertion_N_A" means haf of insertions are A, the other no insertion means haf of insertions are A, the other no insertion.
    """

    def __init__(self):
        self.allele_dict = {}
        class2_remain_list = CLASSES_PROB_2.copy()

        for snv in SNVS:
            key = f"snv_{snv[0]}_{snv[1]}"
            self.allele_dict[key] = (snv[0], snv[1])
        # print(f"number of snvs: {len(self.allele_dict)}, {self.allele_dict}")

        for first in CLASSES_PROB_2:
            for second in class2_remain_list:
                key = f"insertion_{first}_{second}"
                self.allele_dict[key] = (first, second)
            class2_remain_list.remove(first)
        # print(f"total number of genotypes: {len(self.allele_dict)}, {self.allele_dict}")


def process_base(base):
    return base if base == "N" else IUPAC_base_to_ACGT_base_dict[base]


def one_hot_seq(seq):
    vocab_size = len(VOCAB)
    one_hot = np.zeros((len(seq), vocab_size), dtype=np.float32)
    for i, char in enumerate(seq):
        if char in VOCAB:
            one_hot[i, VOCAB[char] - 1] = 1
    return one_hot

def kmer_seq(seq, k=3):
    len_seq = len(seq)
    len_vocab = len(VOCAB_KMER)

    kmer_encodings = []

    if len_seq < k:
        return None
    
    for i in range(len_seq - k + 1):
        kmer = seq[i: i + k]
        kmer_value = 0
        for j, char in enumerate(kmer):
            if char not in VOCAB_KMER:
                return None
            kmer_value += VOCAB_KMER[char] * (len_vocab ** (k - j - 1))

        kmer_encodings.append(kmer_value)

    kmer_encodings = np.array(kmer_encodings, dtype=np.float32)
    return kmer_encodings

