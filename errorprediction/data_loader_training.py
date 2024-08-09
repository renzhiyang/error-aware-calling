import os
import numpy as np

from utils import *

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
            print(f"len array:{len(array)}, array:{array}" f"input seq: {input_seq}")
            return None

        array = np.array(array, dtype=np.float32)
        return array

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
        up_seq = data_dict["seq_around"][1:-1]
        label_1 = data_dict["label"]
        label_2 = "N"

        if data_dict["read_strand"] == "reverse":
            label_1 = reverse_complement(label_1)

        label_array_1, label_array_2 = self.label_tokenization(label_1, label_2)
        input_array = self.input_tokenization(up_seq)
        # return input_array, label_array_1, label_array_2
        return [(input_array, label_array_1, label_array_2)]

    def load_insertion(self, data_dict):
        up_seq = data_dict["seq_around"][:-1]
        type, read_strand = data_dict["type"], data_dict["read_strand"]

        if type == "Positive" and read_strand == "forward":
            label_1 = data_dict["label"][0]
            label_2 = data_dict["label"][1:]
        elif type == "Positive" and read_strand == "reverse":
            label_1 = data_dict["seq_around"][-1]
            label_2 = reverse_complement(data_dict["label"][1:])
        elif type == "Negative" and read_strand == "forward":
            label_1 = data_dict["label"][0]
            label_2 = "N"
        elif type == "Negative" and read_strand == "reverse":
            label_1 = data_dict["seq_around"][-1]
            label_2 = reverse_complement(data_dict["label"][1:])

        if len(label_2) >= 3:
            label_2 = "rep" + str(len(label_2))

        input_array = self.input_tokenization(up_seq)
        label_array_1, label_array_2 = self.label_tokenization(label_1, label_2)
        # return input_array, label_array_1, label_array_2
        return [(input_array, label_array_1, label_array_2)]

    def load_deletion(self, data_dict):
        deletion_samples = []
        type, read_strand = data_dict["type"], data_dict["read_strand"]
        # positive, forward/reverse
        if type == "Positive" and read_strand in ["forward", "reverse"]:
            num_samples = len(data_dict["read_base"]) - 1
            for i in range(num_samples):
                up_seq = data_dict["seq_around"][i + 1 :] + "-" * i
                label_1 = "-"
                label_2 = "N"
                input_array = self.input_tokenization(up_seq)
                label_array_1, label_array_2 = self.label_tokenization(label_1, label_2)
                # print(f'type: {type}, strand: {read_strand}, len input: {len(input_array)}'
                #      f'up seq:{up_seq}, {len(up_seq)}'
                #      f'input:{input_array}, label1:{label_array_1}, label2: {label_array_2}')
                # yield input_array, label_array_1, label_array_2
                deletion_samples.append((input_array, label_array_1, label_array_2))

        # negative, forward
        elif type == "Negative" and read_strand == "forward":
            num_samples = len(data_dict["read_base"]) - 1
            label = data_dict["label"]
            for i in range(num_samples):
                up_seq = data_dict["seq_around"][i + 1 :]
                if i > 0:
                    up_seq = data_dict["seq_around"][i + 1 :] + label[1 : i + 1]
                label_1 = label[i + 1]
                label_2 = "N"

                seq = data_dict["seq_around"]
                input_array = self.input_tokenization(up_seq)
                label_array_1, label_array_2 = self.label_tokenization(label_1, label_2)
                # print(f'type: {type}, strand: {read_strand}, len input: {len(input_array)}'
                #      f'up seq:{up_seq}, {len(up_seq)}'
                #      f'input:{input_array}, label1:{label_array_1}, label2: {label_array_2}')
                # yield input_array, label_array_1, label_array_2
                deletion_samples.append((input_array, label_array_1, label_array_2))

        # negative, reverse
        elif type == "Negative" and read_strand == "reverse":
            num_samples = len(data_dict["read_base"])
            label = reverse_complement(data_dict["label"])
            for i in range(num_samples):
                up_seq = data_dict["seq_around"][i + 1 :]
                if i > 0:
                    up_seq = data_dict["seq_around"][i + 1 :] + label[:i]
                label_1 = label[i]
                label_2 = "N"
                input_array = self.input_tokenization(up_seq)
                label_array_1, label_array_2 = self.label_tokenization(label_1, label_2)
                # print(f'type: {type}, strand: {read_strand}, len input: {len(input_array)}'
                #      f'up seq:{up_seq}, {len(up_seq)}'
                #      f'input:{input_array}, label1:{label_array_1}, label2: {label_array_2}')
                # yield input_array, label_array_1, label_array_2
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
            if len(data_dict["label"]) > 7 and data_dict["variant_type"] == "Insertion":
                return None
            if (
                data_dict["variant_type"] == "SNV"
                and len(data_dict["seq_around"]) != 101
            ):
                return None
            if (
                data_dict["variant_type"] in ["Insertion", "Deletion"]
                and len(data_dict["seq_around"]) != 100
            ):
                return None

            # for details, see https://kalab.docbase.io/posts/3374958
            if data_dict["variant_type"] == "SNV":
                # input_array, label_array_1, label_array_2 = self.load_snv(data_dict)
                # return input_array, label_array_1, label_array_2
                # return None
                input_label_array = self.load_snv(data_dict)

            elif data_dict["variant_type"] == "Insertion":
                # input_array, label_array_1, label_array_2 = self.load_insertion(data_dict)
                # return input_array, label_array_1, label_array_2
                # return None
                input_label_array = self.load_insertion(data_dict)

            elif data_dict["variant_type"] == "Deletion":
                input_label_array = self.load_deletion(data_dict)

            return input_label_array

