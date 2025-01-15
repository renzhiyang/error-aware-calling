# input: one-hot encoding for each base
# output: one-hot encoding for each class

import os
import torch
import mmap
import numpy as np
from numpy.core.multiarray import array
from torch.utils.data import Dataset

import errorprediction.utils as utils


class Data_Loader(Dataset):
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

        self.__count_lines_and_offsets()

    def __count_lines_and_offsets(self):
        with open(self.file_path, "r") as file:
            mmapped_file = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
            offset = 0
            self.line_offsets.append(offset)

            while True:
                line = mmapped_file.readline()
                if not line:
                    break
                offset += len(line)
                self.total_lines += 1
                if self.total_lines % self.chunk_size == 0:
                    self.line_offsets.append(offset)
            mmapped_file.close()
        print(f"Total lines in the file: {self.total_lines}")
        # print(f"offsets: {self.line_offsets}")

    def __len__(self):
        return self.total_lines

    def __load_line(self, line):
        """
        Parse a line of data and return a dictionary containing the parsed values.
        """
        data_dict = {}
        parts = line.strip().split(" ")
        data_dict["chrom"] = parts[0].split(":")[1]
        data_dict["type"] = parts[1].split(":")[1]
        data_dict["read_strand"] = parts[2].split(":")[1]
        data_dict["pos_ref"] = parts[3].split(":")[1]
        data_dict["label"] = parts[4].split(":")[1]
        data_dict["read_base"] = parts[5].split(":")[1]
        data_dict["variant_type"] = parts[-2].split(":")[1]
        data_dict["seq_around"] = parts[-1].split(":")[1]
        return data_dict

    def __getitem__(self, idx):
        chunk_idx = idx // self.chunk_size
        line_idx = idx % self.chunk_size

        with open(self.file_path, "r") as file:
            file.seek(self.line_offsets[chunk_idx])
            for _ in range(line_idx):
                file.readline()

            # load each line
            line = file.readline()
            data_dict = self.__load_line(line)

            # skip insertion length > 6
            if (
                len(data_dict["read_base"]) > 7
                and data_dict["variant_type"] == "Insertion"
            ):
                return None

            # process SNV, Insertion, and Deletion seperately
            if data_dict["variant_type"] == "SNV":
                input_label_array = load_with_kmer_groudTruth_snv(data_dict, self.config)
            elif data_dict["variant_type"] == "Insertion":
                input_label_array = load_with_kmer_groundTruth_insertion(
                    data_dict, self.config
                )
            elif data_dict["variant_type"] == "Deletion":
                # TODO currently exclude deletion, should change the outputs in the future
                return None
                # input_label_array = self.__load_with_kmer_groundTruth_deletion(
                #    data_dict
                # )
            else:
                return None

            return input_label_array


class Data_Loader_Inmemory_pt(Dataset):
    # load a pt file in memory and get data by idx
    def __init__(self, pt_file):
        self.samples = torch.load(pt_file)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class Data_Loader_Inmemory(Dataset):
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

        self.__count_lines_and_offsets()

    def __count_lines_and_offsets(self):
        with open(self.file_path, "r") as file:
            mmapped_file = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
            offset = 0
            self.line_offsets.append(offset)

            while True:
                line = mmapped_file.readline()
                if not line:
                    break
                offset += len(line)
                self.total_lines += 1
                if self.total_lines % self.chunk_size == 0:
                    self.line_offsets.append(offset)
            mmapped_file.close()
        print(f"Total lines in the file: {self.total_lines}")
        # print(f"offsets: {self.line_offsets}")

    def __len__(self):
        return self.total_lines

    def __load_line(self, line):
        """
        Parse a line of data and return a dictionary containing the parsed values.
        """
        data_dict = {}
        parts = line.strip().split(" ")
        data_dict["chrom"] = parts[0].split(":")[1]
        data_dict["type"] = parts[1].split(":")[1]
        data_dict["read_strand"] = parts[2].split(":")[1]
        data_dict["pos_ref"] = parts[3].split(":")[1]
        data_dict["label"] = parts[4].split(":")[1]
        data_dict["read_base"] = parts[5].split(":")[1]
        data_dict["variant_type"] = parts[-2].split(":")[1]
        data_dict["seq_around"] = parts[-1].split(":")[1]
        return data_dict

    def __getitem__(self, idx):
        chunk_idx = idx // self.chunk_size
        line_idx = idx % self.chunk_size

        with open(self.file_path, "r") as file:
            file.seek(self.line_offsets[chunk_idx])
            for _ in range(line_idx):
                file.readline()

            # load each line
            line = file.readline()
            data_dict = self.__load_line(line)

            # skip insertion length > 6
            if (
                len(data_dict["read_base"]) > 7
                and data_dict["variant_type"] == "Insertion"
            ):
                return None

            # process SNV, Insertion, and Deletion seperately
            if data_dict["variant_type"] == "SNV":
                input_label_array = load_with_kmer_groudTruth_snv(data_dict, self.config)
            elif data_dict["variant_type"] == "Insertion":
                input_label_array = load_with_kmer_groundTruth_insertion(
                    data_dict, self.config
                )
            elif data_dict["variant_type"] == "Deletion":
                # TODO currently exclude deletion, should change the outputs in the future
                return None
                # input_label_array = self.__load_with_kmer_groundTruth_deletion(
                #    data_dict
                # )
            else:
                return None

            return input_label_array


def label_tokenization(label_1: str, label_2: str):
    """
    Map Input nucleotide label to array

    Args:
    - label_1 (str): the first label, one of -,A,C,G,T
    - label_2 (str): the second label, one of -,A,C,..,rep6

    Returns:
    - label_array_1: a 1x5 matrix encode the input label_1. e.g., C->[0,0,1,0,0]
    - label_array_2: a 1x25 matrix encode the label_2
    """
    num_classes_prob_1 = len(utils.CLASSES_PROB_1)
    num_classes_prob_2 = len(utils.CLASSES_PROB_2)

    # Initialize arrays to store label data
    label_array_1 = np.zeros(num_classes_prob_1)
    index_label_1 = utils.CLASSES_PROB_1.index(label_1)
    label_array_1[index_label_1] = 1

    label_array_2 = np.zeros(num_classes_prob_2)
    index_label_2 = utils.CLASSES_PROB_2.index(label_2)
    label_array_2[index_label_2] = 1

    return label_array_1, label_array_2

def load_with_kmer_groudTruth_snv(data_dict, config):
    # load data
    pos = data_dict["pos_ref"]
    seq_around = data_dict["seq_around"]
    next_base = data_dict["read_base"]
    read_strand = data_dict["read_strand"]
    label = data_dict["label"]

    # process inputs and labels
    up_seq = seq_around[: config.training.up_seq_len]
    down_seq = seq_around[config.training.up_seq_len + 1 :]
    next_insertion = "-"
    label_1 = label
    label_2 = "-"
    if read_strand == "reverse":
        next_base = utils.reverse_complement(next_base)
        label_1 = utils.reverse_complement(label_1)
    if label_1 == "N":
        label_1 = "-"
    if next_base == "N":
        next_base = "-"

    # include next_base and next_insertion into inputs
    input_array = utils.input_tokenization_include_ground_truth_kmer_context(
        up_seq, down_seq, next_base, next_insertion, config.training.kmer
    )
    label_array_1, label_array_2 = label_tokenization(label_1, label_2)
    if input_array is None:
        return None
    # return [(input_array, label_array_1, label_array_2)]
    return (input_array, label_array_1, label_array_2)

def load_with_kmer_groundTruth_insertion(data_dict, config):
    pos = data_dict["pos_ref"]
    seq_around = data_dict["seq_around"][:-1]
    type, read_strand = data_dict["type"], data_dict["read_strand"]
    read_base = data_dict["seq_around"]
    label = data_dict["label"]

    up_seq = seq_around[: config.training.up_seq_len]
    down_seq = seq_around[config.training.up_seq_len + 1 :]
    next_base = seq_around[config.training.up_seq_len]
    next_insertion = read_base[1:]
    label_1 = label[0]
    label_2 = label[1:]

    if read_strand == "reverse":
        next_insertion = utils.reverse_complement(read_base)[1:]
        label_1 = utils.reverse_complement(label)[0]
        label_2 = utils.reverse_complement(label)[1:]
    if label_2 in ("", "N"):
        label_2 = "-"
    if len(label_2) > 1:
        label_2 = label_2.replace("N", "")

    # exclude smaples with insertion length larger than 6
    if len(label_2) >= 3 and len(label_2) <= 6:
        label_2 = "rep" + str(len(label_2))
    else:
        return None

    if len(next_insertion) >= 3 and len(next_insertion) <= 6:
        insertion = "rep" + str(len(label_2))
    else:
        return None

    # include next_base and next_insertion into inputs
    input_array = utils.input_tokenization_include_ground_truth_kmer_context(
        up_seq, down_seq, next_base, insertion, config.training.kmer
    )

    label_array_1, label_array_2 = label_tokenization(label_1, label_2)
    if input_array is None:
        return None

    # return [(input_array, label_array_1, label_array_2)]
    return (input_array, label_array_1, label_array_2)

def load_with_kmer_groundTruth_deletion(data_dict, config):
    deletion_sampels = []
    type, read_strand, read_base = (
        data_dict["type"],
        data_dict["read_strand"],
        data_dict["read_base"],
    )
    seq_around = data_dict["seq_around"]
    label = data_dict["label"]
    del_len = len(read_base) - 2

    if type == "Positive":
        label = read_base
    if read_strand == "reverse":
        read_base = utils.reverse_complement(read_base)
        label = utils.reverse_complement(label)
    read_base = read_base[:-1]
    label = label[:-1]

    # exclude deletion length larger than window size
    if len(data_dict["read_base"]) - 2 >= config.training.up_seq_len:
        return []

    num_samples = len(read_base)
    for i in range(num_samples):
        up_seq = seq_around[i:] + "-" * i
        next_base = read_base[i]
        insertion = "-"
        label_1 = label[i]
        label_2 = "-"
        # include next_base and next_insertion into inputs
        input_array = utils.input_tokenization_include_ground_truth_kmer(
            up_seq, next_base, insertion, config.training.kmer
        )
        # exclude next_base and next_insertion
        # input_array = utils.input_tokenization_without_grount_truth_kmer(
        #    up_seq, self.config.training.kmer
        # )
        label_array_1, label_array_2 = label_tokenization(label_1, label_2)
        if input_array is not None:
            deletion_sampels.append((input_array, label_array_1, label_array_2))
    return deletion_sampels