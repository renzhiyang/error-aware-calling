import numpy as np

VOCAB = {'A': 1, 'C': 2, 'G': 3, 'T': 4, '-': 5, 'N':6}
CLASSES_PROB_1 = ['A', 'C', 'G', 'T', '-']
CLASSES_PROB_2 = ['N', 'A', 'C', 'G', 'T', 'AA', 'AC', 'AG', 'AT', 'CA',
                    'CC', 'CG', 'CT', 'GA', 'GC', 'GG', 'GT', 'TA', 'TC',
                    'TG', 'TT', 'rep3', 'rep4', 'rep5', 'rep6']

IUPAC_base_to_ACGT_base_dict = dict(zip(
    "ACGTURYSWKMBDHVNP-",
    ("A", "C", "G", "T", "T", "A", "C", "C", "A", "G", "A", "C", "A", "A", "A", "N", "P", "-")
))

def process_base(base):
    return base if base == "N" else IUPAC_base_to_ACGT_base_dict[base]

def one_hot_seq(seq):
    vocab_size = len(VOCAB)
    one_hot = np.zeros((len(seq), vocab_size), dtype=int)
    for i, char in enumerate(seq):
        if char in VOCAB:
            one_hot[i, VOCAB[char] - 1] = 1
    return one_hot