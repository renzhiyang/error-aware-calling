# preprocess label file to pt file which can be used during training
import torch
import argparse
import numpy as np

import errorprediction.utils as utils

from datetime import datetime


def load_line(line):
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
    up_seq = seq_around[: config.up_seq_len]
    down_seq = seq_around[config.up_seq_len + 1 :]
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
        up_seq, down_seq, next_base, next_insertion, config.kmer
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

    up_seq = seq_around[: config.up_seq_len]
    down_seq = seq_around[config.up_seq_len + 1 :]
    next_base = seq_around[config.up_seq_len]
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
        up_seq, down_seq, next_base, insertion, config.kmer
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
    if len(data_dict["read_base"]) - 2 >= config.up_seq_len:
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
            up_seq, next_base, insertion, config.kmer
        )
        # exclude next_base and next_insertion
        # input_array = utils.input_tokenization_without_grount_truth_kmer(
        #    up_seq, self.config.training.kmer
        # )
        label_array_1, label_array_2 = label_tokenization(label_1, label_2)
        if input_array is not None:
            deletion_sampels.append((input_array, label_array_1, label_array_2))
    return deletion_sampels


def preprocess_and_save(config):
    samples = []
    with open(config.label_fn, "r") as f:
        for line in f:
            data_dict = load_line(line)
            
            if (
                len(data_dict["read_base"]) > 7
                and data_dict["variant_type"] == "Insertion"
            ):
                continue
                
            if data_dict["variant_type"] == "SNV":
                input_label_array = load_with_kmer_groudTruth_snv(data_dict, config)
            elif data_dict["variant_type"] == "Insertion":
                input_label_array = load_with_kmer_groundTruth_insertion(
                    data_dict, config
                )
            elif data_dict["variant_type"] == "Deletion":
                #TODO current processing codes is uncorrect
                continue
                input_label_array = load_with_kmer_groundTruth_deletion(
                    data_dict, config
                )
            
            single_result = input_label_array
            if single_result is not None:
                samples.append(single_result)

    torch.save(samples, config.pt_fn)
    print(f"Saved preprocessed data to {config.pt_fn}, total {len(samples)} samples.")


def main():
    parser = argparse.ArgumentParser(description="generate training data")
    parser.add_argument(
        "--label_fn",
        type=str,
        help="label file path",
        default="",
        required=True,
    )
    parser.add_argument(
        "--pt_fn",
        type=str,
        help="pt file path",
        default="",
        required=True,
    )
    parser.add_argument(
        "--kmer",
        type=int,
        help="kmer size",
        default=5,
        required=True,
    )
    parser.add_argument(
        "--up_seq_len",
        type=int,
        help="upstream sequence length",
        default=40,
        required=True,
    )
    args = parser.parse_args()
    preprocess_and_save(args)


if __name__ == "__main__":
    main()
