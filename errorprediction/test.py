import os
import hydra
import torch
import pysam
import numpy as np

from omegaconf import DictConfig, OmegaConf


# @hydra.main(version_base=None, config_path='../../configs/', config_name='defaults.yaml')
def my_app(config: DictConfig) -> None:
    config = config.label_data
    print(OmegaConf.to_yaml(config))
    # print(config.data_path.vcf_f)
    bam = pysam.AlignmentFile(config.data_path.tagged_bam)
    for read in bam:
        if read.is_reverse:
            print(f"forward?: {read.is_forward}")
            print(f"CIGAR:{read.cigar}")
            print(f"reference s:{read.get_reference_sequence()},")
            print(f"forward seq:{read.get_forward_sequence()},")
            print(f"reversed se:{reverse_complement(read.query_sequence)}")
            print(f"query seque:{read.query_sequence} \n")


def reverse_complement(dna_sequence):
    complement = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join(complement[base] for base in dna_sequence[::-1])


def labeled_data_test():
    file = "/home/yang1031/projects/error-aware-calling/data/illumina/label_chr1_100window_phased"
    with open(file, "r") as f:
        for line in f:
            parts = line.strip().split(" ")
            chrom = parts[0].split(":")[1]
            type = parts[1].split(":")[1]
            read_strand = parts[2].split(":")[1]
            pos_ref = parts[3].split(":")[1]
            label = parts[4].split(":")[1]
            read_base = parts[5].split(":")[1]
            ref_base = parts[6].split(":")[1]
            print(parts)


def label_data(label_1, label_2):
    classes_prob_1 = ["A", "C", "G", "T", "-"]
    num_classes_prob_1 = len(classes_prob_1)

    # Define classes for the second set of probabilities
    classes_prob_2 = [
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
    num_classes_prob_2 = len(classes_prob_2)

    # Initialize arrays to store label data
    label_array_1 = np.zeros(num_classes_prob_1)
    index_label_1 = classes_prob_1.index(label_1)
    label_array_1[index_label_1] = 1

    label_array_2 = np.zeros(num_classes_prob_2)
    index_label_2 = classes_prob_2.index(label_2)
    label_array_2[index_label_2] = 1
    print(label_array_1, label_array_2)


def count_matching_indices(tensor1, tensor2):
    # Perform element-wise comparison between tensor1 and tensor2
    matching_indices = tensor1 == tensor2
    print(matching_indices)
    # Count the number of True values (matching indices)
    num_matching_indices = matching_indices.sum()
    return num_matching_indices


if __name__ == "__main__":
    # my_app()
    # labeled_data_test()
    tensor1 = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    tensor2 = torch.tensor([1, 0, 0, 1, 0, 0, 1, 1, 1, 1])
    tensor3 = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    tensor4 = torch.tensor([1, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    count = 0
    for i in range(tensor1.shape[0]):
        if tensor1[i] == tensor2[i] and tensor3[i] == tensor4[i]:
            count += 1
    print(count)
    matching_indices_next_base = count_matching_indices(tensor1, tensor2)
    matching_indices_insertion = count_matching_indices(tensor3, tensor4)
    matching_total = count_matching_indices(
        matching_indices_next_base, matching_indices_insertion
    )
    print(
        "Number of matching indices between next_base and true_next_base:",
        matching_indices_next_base,
    )
    print(
        "Number of matching indices between insertion and true_insertion:",
        matching_indices_insertion,
    )
    print(
        "Number of matching indices between insertion and true_insertion:",
        matching_total,
    )

