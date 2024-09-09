# input: one-hot encoding for each base
# output: one-hot encoding for each class

import os
import numpy as np

from errorprediction.utils import *

VOCAB = {"A": 1, "C": 2, "G": 3, "T": 4, "-": 5, "N": 6}
VOCAB_KMER = {"N": 0, "A": 1, "C": 2, "G": 3, "T": 4, "-": 5}
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


class Data_Loader:
    def __init__(self, file_path, config, chunk_size=1000):
        """
        Initialize the Data_Loader object with file_path, chunk_size, and configuration.
        """
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.line_offsets = []
        self.total_lines = 0
        self.config = config

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist")

        with open(file_path, "r") as file:
            offset = 0
            for line in file:
                if self.total_lines % self.chunk_size == 0:
                    self.line_offsets.append(offset)
                offset += len(line)
                self.total_lines += 1
        print(self.total_lines)

    def __len__(self):
        return self.total_lines

    def load_line(self, line):
        """
        Parse a line of data and return a dictionary containing the parsed values.
        """
        data_dict = {}
        parts = line.strip().split(" ")
        data_dict["chrom"] = parts[0].split(":")[1]
        data_dict["type"] = parts[1].split(":")[1]
        data_dict["read_strand"] = parts[2].split(":")[1]
        data_dict["pos"] = parts[3].split(":")[1]
        data_dict["label"] = parts[4].split(":")[1]
        data_dict["read_base"] = parts[5].split(":")[1]
        data_dict["variant_type"] = parts[-2].split(":")[1]
        data_dict["seq_around"] = parts[-1].split(":")[1]
        return data_dict

    def input_tokenization(self, input_seq: str):
        """
        Encode input sequence
        """
        # 排除掉不足99个base的
        array = [VOCAB[char] for char in input_seq]
        if len(array) != self.config.training.up_seq_len:
            # print(f'len array:{len(array)}, array:{array}'
            #      f'input seq: {input_seq}')
            return None

        array = np.array(array, dtype=np.float32)
        # print(f'input no ont-hot: {array.shape}')
        return array

    def input_tokenization_onehot(self, input_seq: str):
        """
        Encode input sequence
        """
        len_vocab = len(VOCAB)
        # 排除掉不足99个base的
        array = [VOCAB[char] for char in input_seq]
        if len(array) != self.config.training.up_seq_len:
            # print(f'len array:{len(array)}, array:{array}'
            #      f'input seq: {input_seq}')
            return None

        one_hot_array = []
        for value in array:
            if value == 1:
                one_hot_array.append(np.zeros(len_vocab))
            else:
                one_hot_array.append(np.eye(len_vocab)[int(value) - 1])
        one_hot_array = np.array(one_hot_array, dtype=np.float32)
        # print(f'one-hot array shape: {one_hot_array.shape}')
        return one_hot_array

    def input_tokenization_kmer(self, input_seq: str, k=3):
        """
        Encode input sequence with k-mer
        """
        len_seq = len(input_seq)
        len_vocab = len(VOCAB_KMER)

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
                if char not in VOCAB_KMER:
                    return None  # Handle unkwon characters
                # Calculate the unique value for the k-mer
                kmer_value += VOCAB_KMER[char] * (len_vocab ** (k - j - 1))

            # Append the encoded k-mer value to the list
            kmer_encodings.append(kmer_value)

        # Convert the list to a numpy array for efficient processing
        kmer_encodings = np.array(kmer_encodings, dtype=np.float32)
        return kmer_encodings

    def label_tokenization(self, label_1: str, label_2: str):
        """
        Map Input nucleotide label to array

        Args:
        - label_1 (str): the first label, one of A,C,G,T,-
        - label_2 (str): the second label, one of N,A,C,..,rep6

        Returns:
        - label_array_1: a 1x5 matrix encode the input label_1. e.g., C->[0,1,0,0,0]
        - label_array_2: a 1x25 matrix encode the label_2
        """
        num_classes_prob_1 = len(CLASSES_PROB_1)
        num_classes_prob_2 = len(CLASSES_PROB_2)

        # Initialize arrays to store label data
        label_array_1 = np.zeros(num_classes_prob_1)
        index_label_1 = CLASSES_PROB_1.index(label_1)
        label_array_1[index_label_1] = 1

        label_array_2 = np.zeros(num_classes_prob_2)
        index_label_2 = CLASSES_PROB_2.index(label_2)
        label_array_2[index_label_2] = 1

        return label_array_1, label_array_2

    def load_snv(self, data_dict):
        up_seq = data_dict["seq_around"][:-1]
        label_1 = data_dict["read_base"]
        label_2 = "N"

        if data_dict["read_strand"] == "reverse":
            label_1 = reverse_complement(label_1)
        if label_1 == "N":
            label_1 = "-"

        # print(data_dict['pos'], label_1, label_2)
        label_array_1, label_array_2 = self.label_tokenization(label_1, label_2)
        # input_array = self.input_tokenization(up_seq) # without one hot encoding
        # input_array = self.input_tokenization_onehot(up_seq)  # with one hot encoding
        input_array = self.input_tokenization_kmer(up_seq, k=3)
        # return input_array, label_array_1, label_array_2
        return [(input_array, label_array_1, label_array_2)]

    def load_insertion(self, data_dict):
        up_seq = data_dict["seq_around"][:-1]
        type, read_strand = data_dict["type"], data_dict["read_strand"]

        # TODO: 这里需要对中间输出文件进行修改，reverse strand中的Label之后要换成read base
        label_1 = data_dict["read_base"][0]
        label_2 = data_dict["read_base"][1:]

        # TODO: don't know why now
        # A bug, sometimes label_2 is "AN" or some string with "N"
        if len(label_2) > 1:
            label_2 = label_2.replace("N", "")

        if read_strand == "reverse":
            label_1 = data_dict["seq_around"][-1]
            label_2 = reverse_complement(label_2)

        if len(label_2) >= 3:
            label_2 = "rep" + str(len(label_2))

        # input_array = self.input_tokenization(up_seq)
        # input_array = self.input_tokenization_onehot(up_seq)
        input_array = self.input_tokenization_kmer(up_seq, k=3)
        label_array_1, label_array_2 = self.label_tokenization(label_1, label_2)
        return [(input_array, label_array_1, label_array_2)]

    def load_deletion(self, data_dict):
        deletion_samples = []
        type, read_strand, read_base = (
            data_dict["type"],
            data_dict["read_strand"],
            data_dict["read_base"],
        )
        seq_around = data_dict["seq_around"][1:]
        read_base = read_base[:-1]

        if read_strand == "reverse":
            read_base = reverse_complement(read_base)

        num_samples = len(read_base)
        for i in range(num_samples):
            up_seq = seq_around[i:] + "-" * i
            label_1 = read_base[i]
            label_2 = "N"
            # input_array = self.input_tokenization(up_seq)
            # input_array = self.input_tokenization_onehot(up_seq)
            input_array = self.input_tokenization_kmer(up_seq, k=3)
            label_array_1, label_array_2 = self.label_tokenization(label_1, label_2)
            deletion_samples.append((input_array, label_array_1, label_array_2))
        return deletion_samples

    def __getitem__(self, idx):
        chunk_idx = idx // self.chunk_size
        line_idx = idx % self.chunk_size

        with open(self.file_path, "r") as file:
            file.seek(self.line_offsets[chunk_idx])
            for _ in range(line_idx):
                file.readline()

            # load each line
            line = file.readline()
            data_dict = self.load_line(line)

            # skip insertion length > 6
            if (
                len(data_dict["read_base"]) > 7
                and data_dict["variant_type"] == "Insertion"
            ):
                return None
            if (
                data_dict["variant_type"] in ["SNV", "Insertion", "Deletion"]
                and len(data_dict["seq_around"]) != 100
            ):
                return None

            # for details, see https://kalab.docbase.io/posts/3374958
            if data_dict["variant_type"] == "SNV":
                input_label_array = self.load_snv(data_dict)

            elif data_dict["variant_type"] == "Insertion":
                input_label_array = self.load_insertion(data_dict)

            elif data_dict["variant_type"] == "Deletion":
                input_label_array = self.load_deletion(data_dict)

            return input_label_array
