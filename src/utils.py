import numpy as np

VOCAB = {"A": 1, "C": 2, "G": 3, "T": 4, "-": 5, "N": 6}
CLASSES_PROB_1 = ["A", "C", "G", "T", "-"]
CLASSES_PROB_2 = [
    "N",
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
    def __init__(self):
        self.allele_dict = {}
        for snv in SNVS:
            key = f"snv_{snv[0]}_{snv[1]}"
            self.allele_dict[key] = (snv[0], snv[1])
        for first in CLASSES_PROB_1:
            if first == "-":
                continue
            for second in CLASSES_PROB_2:
                if second == "N":
                    continue
                key = f"insertion_{first}_{second}"
                self.allele_dict[key] = (first, second)


def process_base(base):
    return base if base == "N" else IUPAC_base_to_ACGT_base_dict[base]


def one_hot_seq(seq):
    vocab_size = len(VOCAB)
    one_hot = np.zeros((len(seq), vocab_size), dtype=np.float32)
    for i, char in enumerate(seq):
        if char in VOCAB:
            one_hot[i, VOCAB[char] - 1] = 1
    return one_hot
