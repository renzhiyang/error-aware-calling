#! /usr/bin/env python3

import os
import torch
import argparse

import src.bayesian as bayesian
import errorprediction.models.nets as nets
import src.utils as utils
import numpy as np

from errorprediction.models.baseline import Baseline
from torch.utils.data import DataLoader


class TensorData:
    def __init__(self, file_path):
        self.file_path = file_path
        self.chunk_size = 1000
        self.line_offset = []
        self.total_lines = 0

        with open(file_path, "r") as f:
            offset = 0
            for line in f:
                self.line_offset.append(offset)
                offset += len(line)
                self.total_lines += 1

    def __len__(self):
        return self.total_lines

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_lines:
            raise IndexError("Index out of range")

        with open(self.file_path, "r") as f:
            # print(f"idx: {idx}, line_offset: {self.line_offset[idx]}")
            f.seek(self.line_offset[idx])
            line = f.readline().strip().split("\t")

            id = line[0]
            input_seq = line[1]
            observe_b = str(line[2])
            observe_ins = str(line[3])

            position = int(id.split("_")[1])
            index = int(id.split("_")[2])

            tensor_kmer = utils.kmer_seq(input_seq, k=3)
            return position, index, tensor_kmer, observe_b, observe_ins


def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor


def predict(args):
    model_fn = args.model
    tensor_fn = args.tensor_fn

    # check file path
    if not os.path.exists(model_fn):
        print(f"Error: Model file '{model_fn}' does not exist.")
        return
    if not os.path.isfile(tensor_fn):
        print(f"Error: Tensor file '{tensor_fn}' does not exist.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if torch.device("mps"):
    #    device = torch.device("mps")

    print(f"device: {device}")
    # model = Baseline().to(device)
    model = nets.Encoder_Transformer(
        embed_size=54,
        vocab_size=216,
        num_layers=1,
        forward_expansion=1024,
        seq_len=97,
        dropout_rate=0.1,
        num_class1=5,
        num_class2=25,
    ).to(device)
    model.load_state_dict(torch.load(model_fn, map_location=device))
    model.eval()

    dataset = TensorData(tensor_fn)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  # type: ignore
    caller = bayesian.BayesianCaller()
    pos_probs_in_pos = {}  # key:position, value: all reads' posterior probs dict
    # count = 0
    for position, idx, tensor_one_hot, observe_b, observe_ins in dataloader:
        # test codes
        # count += 1
        # if count >= 1000:
        # break
        # if position != 7232171:
        #    continue

        tensor_one_hot = tensor_one_hot.float().to(device)
        observe_b = observe_b[0]
        observe_ins = observe_ins[0]
        next_base_dis, insertion_dis = model(tensor_one_hot)

        # test codes
        # print(f"min max base: {next_base_dis.min()}, {next_base_dis.max()}")
        # print(f"min max ins: {insertion_dis.min()}, {insertion_dis.max()}")
        # print(f"position:{position}, tensor:{tensor_one_hot}")
        # print(f"position:{position}, observe_b:{observe_b}, observe_ins:{observe_ins}")

        position = position.numpy().item()
        next_base_dis = next_base_dis.cpu().detach().numpy().reshape(-1)
        insertion_dis = insertion_dis.cpu().detach().numpy().reshape(-1)

        next_base_dis = normalize_tensor(next_base_dis)
        insertion_dis = normalize_tensor(insertion_dis)

        # test codes
        # print(f"observed base:{observe_b}, next base dis:{next_base_dis}")

        if position not in pos_probs_in_pos:
            pos_probs_in_pos[position] = []

        all_genotype_pos_probs_one_read = (
            caller.all_genotypes_posterior_prob_per_read_log(
                next_base_dis, insertion_dis, observe_b, observe_ins
            )
        )

        if len(pos_probs_in_pos[position]) == 0:
            pos_probs_in_pos[position] = all_genotype_pos_probs_one_read
        else:
            pos_probs_in_pos[position] = caller.multiply_pos_probs_of_two_reads(
                pos_probs_in_pos[position], all_genotype_pos_probs_one_read
            )

    for pos, pos_probs in pos_probs_in_pos.items():
        sorted_probs = sorted(pos_probs.items(), key=lambda x: x[1], reverse=True)
        print(f"position: {pos}, most likelily genotype: {next(iter(sorted_probs))}")


def predict_kmer(args):
    model_fn = args.model
    tensor_fn = args.tensor_fn

    # check file path
    if not os.path.exists(model_fn):
        print(f"Error: Model file '{model_fn}' does not exist.")
        return
    if not os.path.isfile(tensor_fn):
        print(f"Error: Tensor file '{tensor_fn}' does not exist.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_f = open(tensor_fn, "r")
    caller = bayesian.BayesianCaller()

    print(f"device: {device}")
    # model = Baseline().to(device)
    model = nets.Encoder_Transformer(
        embed_size=54,
        vocab_size=216,
        num_layers=1,
        forward_expansion=1024,
        seq_len=97,
        dropout_rate=0.1,
        num_class1=5,
        num_class2=25,
    ).to(device)
    model.load_state_dict(torch.load(model_fn, map_location=device))
    model.eval()

    cur_pos = 0
    cur_probs = {}
    for line in tensor_f:
        line = line.strip().split()

        id = line[0]
        input_seq = line[1]
        observe_b = str(line[2])
        observe_ins = str(line[3])

        position = int(id.split("_")[1])
        index = int(id.split("_")[2])

        if cur_pos != position:
            if cur_pos != 0:
                sorted_probs = sorted(
                    cur_probs.items(), key=lambda x: x[1], reverse=True
                )
                print(
                    f"position: {cur_pos}, most likelily genotype: {sorted_probs[:5]}"
                )
            cur_pos = position
            cur_probs = []

        tensor_kmer = utils.kmer_seq(input_seq, k=3)
        tensor_kmer = torch.tensor(tensor_kmer).float().unsqueeze(0).to(device)
        next_base_dis, insertion_dis = model(tensor_kmer)
        next_base_dis = next_base_dis.cpu().detach().numpy().reshape(-1)
        insertion_base_dis = insertion_dis.cpu().detach().numpy().reshape(-1)

        next_base_dis = normalize_tensor(next_base_dis)
        insertion_base_dis = normalize_tensor(insertion_base_dis)

        all_genotype_pos_probs_one_read = (
            caller.all_genotypes_posterior_prob_per_read_log(
                next_base_dis, insertion_base_dis, observe_b, observe_ins
            )
        )

        if len(cur_probs) == 0:
            cur_probs = all_genotype_pos_probs_one_read
        else:
            cur_probs = caller.multiply_pos_probs_of_two_reads(
                cur_probs, all_genotype_pos_probs_one_read
            )


def main():
    parser = argparse.ArgumentParser(
        description="call variants using fine-turned model"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="the fine-turned model that used for prediction",
        required=True,
    )
    parser.add_argument(
        "--tensor_fn", type=str, help="tensor file generate by generate_tensor.py"
    )
    args = parser.parse_args()
    # predict(args)
    predict_kmer(args)


if __name__ == "__main__":
    main()
