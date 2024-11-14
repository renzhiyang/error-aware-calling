#! /usr/bin/env python3
import os
from regex import W
import yaml
import torch
import argparse
import numpy as np

import src.utils as utils
import src.bayesian as bayesian
import errorprediction.models.nets as nets
import errorprediction.utils as model_utils

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

            # tensor_kmer = utils.kmer_seq(input_seq, k=3)
            return position, index, input_seq, observe_b, observe_ins


def load_model_paras(config_f: str):
    # input is model parameters' config yaml file
    if not os.path.exists(config_f):
        print("config file path not exists")

    config = open(config_f, "r")
    config = yaml.safe_load(config)
    params = config["training"]
    return params


def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    # resize all probs to (0,1), and sum to 1
    # min_val = tensor.min()
    # max_val = tensor.max()
    # normalized_tensor = (tensor - min_val) / (max_val - min_val)
    # return normalized_tensor / normalized_tensor.sum()
    shift_tensor = tensor - tensor.min() + 1  # +1 to avoid 0 probs
    normalized_tensor = shift_tensor / shift_tensor.sum()
    return normalized_tensor


def reverse_prob_distributions(next_base_dis, insertion_dis):
    # change the order of A,C,G,T for reverse strand
    next_base_dis[1], next_base_dis[-1] = next_base_dis[-1], next_base_dis[1]
    next_base_dis[2], next_base_dis[3] = next_base_dis[3], next_base_dis[2]
    # TODO change the order of insertion distribution

    return next_base_dis, insertion_dis


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


def print_genotype(cur_pos, cur_probs, out_fn):
    snv_probs = {k: v for k, v in cur_probs.items() if k.startswith("snv")}
    ins_probs = {k: v for k, v in cur_probs.items() if k.startswith("insertion")}

    snv_sorted = sorted(snv_probs.items(), key=lambda x: x[1], reverse=True)
    ins_sorted = sorted(ins_probs.items(), key=lambda x: x[1], reverse=True)
    outline = f"position: {cur_pos}  snv: {snv_sorted[:5]}  ins: {ins_sorted[:5]} \n"
    out_fn.write(outline)
    print(outline, flush=True)


def predict_kmer(args):
    model_fn = args.model
    tensor_fn = args.tensor_fn
    out_fn = args.output_fn

    # check file path
    if not os.path.exists(model_fn):
        print(f"Error: Model file '{model_fn}' does not exist.")
        return
    if not os.path.isfile(tensor_fn):
        print(f"Error: Tensor file '{tensor_fn}' does not exist.")
        return

    # new predicted results file of current tensors
    out_fn = open(out_fn, "w")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_f = open(tensor_fn, "r")
    caller = bayesian.BayesianCaller()
    params = load_model_paras(args.config_fn)

    # calculate the kmer vocabulary size
    # [0:kmer_token_shift] values encode class values and [SEP] symbol
    VOCAB_SIZE = params["num_tokens"] ** params["kmer"] + params["kmer_token_shift"]

    print(f"device: {device}")
    # model = Baseline().to(device)
    model = nets.Encoder_Transformer(
        embed_size=params["embed_size"],
        vocab_size=VOCAB_SIZE,
        num_layers=params["num_layers"],
        forward_expansion=params["forward_expansion"],
        seq_len=params["up_seq_len"] - params["kmer"] + 4,  # with [sep] symbol
        # seq_len=params["up_seq_len"] - params["kmer"] + 3,  # without [sep] symbol
        dropout_rate=params["drop_out"],
        num_class1=params["num_class_1"],
        num_class2=params["num_class_2"],
    ).to(device)
    model.load_state_dict(torch.load(model_fn, map_location=device, weights_only=True))
    model.eval()

    cur_pos = 0
    cur_probs = []

    with torch.no_grad():
        for line in tensor_f:
            line = line.strip().split()

            ctg_name = line[0]
            id = line[1]
            input_seq = line[2]
            observe_b = str(line[3])
            observe_ins = str(line[4])
            read_strand = line[5]
            observe_ins = "-" if observe_ins == "N" else observe_ins

            # TODO for some case, label_2 is like "AN"
            # should fix this error
            if "N" in observe_ins:
                continue

            position = int(id.split("_")[1])

            # if position != 29855876:
            #  position != 29948200:
            #     continue

            if cur_pos != position:
                if cur_pos != 0:
                    print_genotype(cur_pos, cur_probs, out_fn)
                cur_pos = position
                cur_probs = []

            input_tokenization = (
                model_utils.input_tokenization_include_ground_truth_kmer(
                    input_seq, observe_b, observe_ins, params["kmer"]
                )
            )
            # tensor_kmer = utils.kmer_seq(input_seq, k=params["kmer"])
            input_tensor = (
                torch.tensor(input_tokenization).float().unsqueeze(0).to(device)
            )
            next_base_dis, insertion_dis = model(input_tensor)
            next_base_dis = next_base_dis.cpu().detach().numpy().reshape(-1)
            insertion_base_dis = insertion_dis.cpu().detach().numpy().reshape(-1)

            next_base_dis = normalize_tensor(next_base_dis)
            insertion_base_dis = normalize_tensor(insertion_base_dis)

            # for reverse strand, change the order of prob distributions
            if read_strand == "reverse":
                # in the genotype calculate, it should use forward-based nucleotide
                observe_b = model_utils.reverse_complement(observe_b)
                # TODO include insertion
                # observe_ins = model_utils.reverse_complement(observe_ins)

                next_base_dis, insertion_base_dis = reverse_prob_distributions(
                    next_base_dis, insertion_base_dis
                )

            # print test
            max_prob_base = utils.CLASSES_PROB_1[np.argmax(next_base_dis)]
            #print(
            #    f"position: {position}, {read_strand}, {input_seq}, observed base(forward-based):{observe_b}, max prob base(foward-based):{max_prob_base}, predicted dis:{next_base_dis} ",
            #    flush=True,
            #)

            all_genotype_pos_probs_one_read = (
                caller.all_genotypes_posterior_prob_per_read_log(
                    next_base_dis, insertion_base_dis, observe_b, observe_ins
                )
            )
            # print(f"probs: {all_genotype_pos_probs_one_read}")
            """
            all_genotype_pos_probs_one_read = (
                caller.all_genotypes_posterior_prob_per_read_log_uniform_prior(
                    next_base_dis, insertion_base_dis, observe_b, observe_ins
                )
            )
            print(f'position: {position}, initial probs: {all_genotype_pos_probs_one_read}')
            """
            if len(cur_probs) == 0:
                cur_probs = all_genotype_pos_probs_one_read
            else:
                cur_probs = caller.add_pos_probs_of_two_reads(
                    cur_probs, all_genotype_pos_probs_one_read
                )
                # print(f"position: {cur_pos}, probs: {cur_probs}")

    #print_genotype(cur_pos, cur_probs, out_fn)


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
        "--tensor_fn",
        type=str,
        help="tensor file generate by generate_tensor.py",
        required=True,
    )
    parser.add_argument(
        "--config_fn",
        type=str,
        help="parameters of model, restored in yaml file",
        required=True,
    )
    parser.add_argument(
        "--output_fn",
        type=str,
        help="the output file, finnaly it will be a VCF file",
    )
    args = parser.parse_args()
    # predict(args)
    predict_kmer(args)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
