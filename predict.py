#! /usr/bin/env python3
import os
from regex import W
import yaml
import math
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


def normalize_genotype(genotype_prob, top: int):
    # genotype_prob are like:  snv: [('snv_C_C', -94.84156544749021), ...]
    # all values are negative
    # return the top genotypes probs and normalize them to [0,1]
    genotype_prob = genotype_prob[:top]
    # probs = [value for _, value in genotype_prob]
    # min_prob = probs[-1]
    # shift_probs = [(value - min_prob) + 1 for value in probs]
    # total_probs = sum(shift_probs)
    # normalize_probs = [shift_prob / total_probs for shift_prob in shift_probs]

    # linear scaling with clipping
    log_probs = [prob for _, prob in genotype_prob]
    min_prob = min(log_probs)
    max_prob = max(log_probs)
    normalize_probs = []
    for prob in log_probs:
        prob = (prob - min_prob) / (max_prob - min_prob + 0.1)  # plus 0.1 to avoid be 0
        prob = 0.1 + 0.8 * prob
        normalize_probs.append(prob)
    normalize_probs = [prob / sum(normalize_probs) for prob in normalize_probs]

    # softmax normalize genotypes
    # probs = [math.exp(value) for _, value in genotype_prob]
    # normalize_probs = [prob / sum(probs) for prob in probs]

    # sigmoid normalize genotypes
    # probs = [1/(1+math.exp(-prob)) for _, prob in genotype_prob]
    # normalize_probs = probs / np.sum(probs)

    for i in range(len(normalize_probs)):
        genotype_prob[i] = (genotype_prob[i][0], normalize_probs[i])
    return genotype_prob


def reverse_prob_distributions(next_base_dis, insertion_dis):
    # change the order of A,C,G,T for reverse strand
    next_base_dis[1], next_base_dis[-1] = next_base_dis[-1], next_base_dis[1]
    next_base_dis[2], next_base_dis[3] = next_base_dis[3], next_base_dis[2]
    # TODO change the order of insertion distribution

    return next_base_dis, insertion_dis


def sort_genotype_by_type_from(genotype_probs, variant_type):
    # sort the probs of genotypes by variant_type ("snv", or "insertion")
    type_probs = {k: v for k, v in genotype_probs.items() if k.startswith(variant_type)}
    type_probs_sorted = sorted(type_probs.items(), key=lambda x: x[1], reverse=True)
    return type_probs_sorted


def print_genotype(ctg_name, cur_pos, cur_probs, bayesian_threshold, out_fn):
    snv_sorted = sort_genotype_by_type_from(cur_probs, "snv")
    ins_sorted = sort_genotype_by_type_from(cur_probs, "insertion")

    if len(snv_sorted) == 0 or len(ins_sorted) == 0:
        return

    snv_sorted = normalize_genotype(snv_sorted, 5)
    ins_sorted = normalize_genotype(ins_sorted, 5)
    if snv_sorted[0][1] < bayesian_threshold:
        # filter genotype with prob smaller than threshold
        return
    outline = (
        f"ctg:{ctg_name}  position:{cur_pos}  snv:{snv_sorted}  ins:{ins_sorted} \n"
    )
    out_fn.write(outline)
    print(outline, flush=True)


def print_genotype_by_strand(
    caller,
    ctg_name,
    cur_pos,
    cur_probs_forward,
    cur_probs_reverse,
    bayesian_threshold,
    out_fn,
):
    combined_probs = caller.add_pos_probs_of_two_reads(
        cur_probs_forward, cur_probs_reverse
    )
    # if there are no forward or reverse strand reads, print
    if len(cur_probs_forward) == 0 or len(cur_probs_reverse) == 0:
        print_genotype(ctg_name, cur_pos, combined_probs, bayesian_threshold, out_fn)
        return

    snv_forward_sorted = sort_genotype_by_type_from(cur_probs_forward, "snv")
    ins_forward_sorted = sort_genotype_by_type_from(cur_probs_forward, "insertion")
    snv_reverse_sorted = sort_genotype_by_type_from(cur_probs_reverse, "snv")
    ins_reverse_sorted = sort_genotype_by_type_from(cur_probs_reverse, "insertion")
    snv_combined_sorted = sort_genotype_by_type_from(combined_probs, "snv")
    snv_forward_max_genotype = snv_forward_sorted[0][0]
    snv_reverse_max_genotype = snv_reverse_sorted[0][0]
    if snv_forward_max_genotype != snv_reverse_max_genotype:
        if "-" not in snv_forward_max_genotype and "-" not in snv_reverse_max_genotype:
            print(
                f"{cur_pos}, forward: {snv_forward_sorted[0]}, reverse: {snv_reverse_sorted[0]}, total: {snv_combined_sorted[0]}"
            )
    pass


def predict_bayesian2(args):
    model_fn = args.model
    tensor_fn = args.tensor_fn
    out_fn = args.output_fn

    # print button
    is_print_test = True

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
    likelihoods = {}

    with torch.no_grad():
        base_count = {base: 0 for base in utils.CLASSES_PROB_1}
        ins_count = {ins: 0 for ins in utils.CLASSES_PROB_2}
        for line in tensor_f:
            line = line.strip().split()

            ctg_name = line[0]
            id = line[1]
            input_seq = line[2]
            observe_b = str(line[3])
            observe_ins = str(line[4])
            read_strand = line[5]
            observe_ins = "-" if observe_ins == "N" else observe_ins
            position = int(id.split("_")[1])

            if read_strand == "reverse":
                observe_b = model_utils.reverse_complement(observe_b)
                # TODO for inseriont

            if is_print_test:
                if position != 29940059:
                    continue

            # TODO for some case, label_2 is like "AN"
            # should fix this error
            if "N" in observe_ins:
                continue

            if cur_pos != position:
                if cur_pos != 0:
                    #TODO currently the prob distribution for '-' is uncorrect
                    # So lets filter them now.
                    if base_count['-'] == 0: 
                        print(f"base count: {base_count}")
                        # print the posterior probability of genotype
                        prior_genotypes = caller.prior_probability_of_genotype_log(
                            base_count, ins_count
                        )
                        pos_probs = {
                            key: prior_genotypes[key] + likelihoods[key]
                            for key in prior_genotypes
                        }

                        print_genotype(
                            ctg_name, cur_pos, pos_probs, args.bayesian_threshold, out_fn
                        )
                    # reset counts
                    base_count = {base: 0 for base in base_count}
                    ins_count = {ins: 0 for ins in ins_count}

                cur_pos = position
                likelihoods = {}

            # update base_count and ins_count
            base_count[observe_b] += 1
            ins_count[observe_ins] += 1

            # tokenize input sentence
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

            all_genotype_pos_probs_one_read = (
                caller.all_genotypes_likelihood_per_read_log(
                    next_base_dis, insertion_base_dis, observe_b, observe_ins
                )
            )

            if len(likelihoods) == 0:
                likelihoods = all_genotype_pos_probs_one_read
            else:
                likelihoods = caller.add_pos_probs_of_two_reads(
                    likelihoods, all_genotype_pos_probs_one_read
                )

            # print test
            if is_print_test:
                max_prob_base = utils.CLASSES_PROB_1[np.argmax(next_base_dis)]
                cur_genos = sort_genotype_by_type_from(all_genotype_pos_probs_one_read, "snv")
                sorted_likelihoods = sort_genotype_by_type_from(likelihoods, "snv")
                # top_5 = normalize_genotype(sorted_likelihoods, 5)
                np.set_printoptions(precision=4)
                print(
                    f"position: {position}, {read_strand}, {input_seq}, observed base(forward-based):{observe_b}, max base: {max_prob_base}, predicted dis:{next_base_dis} ",
                    flush=True,
                )
                print(f"cur      genotypes      log: {cur_genos}")
                print(f"accumulated likelihoods log: {sorted_likelihoods}\n")
                # print(f"cur genotys: {normalize_genotype(cur_genos, 14)}")
                # print(f"likelihoods: {normalize_genotype(sorted_likelihoods, 14)}\n")

    if is_print_test:
        print(f"base count: {base_count}")
        prior_genotypes = caller.prior_probability_of_genotype_log(
            base_count, ins_count
        )
        pos_probs = {
            key: prior_genotypes[key] + likelihoods[key] for key in prior_genotypes
        }
        print_genotype("chr21", cur_pos, likelihoods, args.bayesian_threshold, out_fn)


def predict_bayesian1(args):
    model_fn = args.model
    tensor_fn = args.tensor_fn
    out_fn = args.output_fn

    # print button
    is_print_test = True

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
    cur_probs = {}

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

            if is_print_test:
                if position != 29779339:
                    continue

            if cur_pos != position:
                if cur_pos != 0:
                    print_genotype(
                        ctg_name, cur_pos, cur_probs, args.bayesian_threshold, out_fn
                    )
                cur_pos = position
                cur_probs = {}

            input_tokenization = (
                model_utils.input_tokenization_include_ground_truth_kmer(
                    input_seq, observe_b, observe_ins, params["kmer"]
                )
            )
            input_tensor = (
                torch.tensor(input_tokenization).float().unsqueeze(0).to(device)
            )
            next_base_dis, insertion_dis = model(input_tensor)
            next_base_dis = next_base_dis.cpu().detach().numpy().reshape(-1)
            insertion_base_dis = insertion_dis.cpu().detach().numpy().reshape(-1)

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
            if is_print_test:
                max_prob_base = utils.CLASSES_PROB_1[np.argmax(next_base_dis)]
                np.set_printoptions(precision=4)
                print(
                    f"position: {position}, {read_strand}, {input_seq}, observed base(forward-based):{observe_b}, max_base: {max_prob_base}, predicted dis:{next_base_dis} ",
                    flush=True,
                )

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

    if is_print_test:
        print_genotype("chr21", cur_pos, cur_probs, args.bayesian_threshold, out_fn)


def predict_bayesian1_correct_error_within_one_strand(args):
    model_fn = args.model
    tensor_fn = args.tensor_fn
    out_fn = args.output_fn

    # print button
    is_print_test = False

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
    cur_probs_forward = {}
    cur_probs_reverse = {}

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

            if is_print_test:
                if position != 29859217:
                    continue

            if cur_pos != position:  # output posterior probs
                if cur_pos != 0:
                    print_genotype_by_strand(
                        caller,
                        ctg_name,
                        cur_pos,
                        cur_probs_forward,
                        cur_probs_reverse,
                        args.bayesian_threshold,
                        out_fn,
                    )
                cur_pos = position
                cur_probs_forward = {}
                cur_probs_reverse = {}

            input_tokenization = (
                model_utils.input_tokenization_include_ground_truth_kmer(
                    input_seq, observe_b, observe_ins, params["kmer"]
                )
            )
            input_tensor = (
                torch.tensor(input_tokenization).float().unsqueeze(0).to(device)
            )
            next_base_dis, insertion_dis = model(input_tensor)
            next_base_dis = next_base_dis.cpu().detach().numpy().reshape(-1)
            insertion_base_dis = insertion_dis.cpu().detach().numpy().reshape(-1)

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
            if is_print_test:
                max_prob_base = utils.CLASSES_PROB_1[np.argmax(next_base_dis)]
                np.set_printoptions(precision=4)
                print(
                    f"position: {position}, {read_strand}, {input_seq}, observed base(forward-based):{observe_b}, predicted dis:{next_base_dis} ",
                    flush=True,
                )

            all_genotype_pos_probs_one_read = (
                caller.all_genotypes_posterior_prob_per_read_log(
                    next_base_dis, insertion_base_dis, observe_b, observe_ins
                )
            )

            # calcurate the likelihood of genotypes by read strand
            if read_strand == "forward":
                if len(cur_probs_forward) == 0:
                    cur_probs_forward = all_genotype_pos_probs_one_read
                else:
                    cur_probs_forward = caller.add_pos_probs_of_two_reads(
                        cur_probs_forward, all_genotype_pos_probs_one_read
                    )
            else:
                if len(cur_probs_reverse) == 0:
                    cur_probs_reverse = all_genotype_pos_probs_one_read
                else:
                    cur_probs_reverse = caller.add_pos_probs_of_two_reads(
                        cur_probs_reverse, all_genotype_pos_probs_one_read
                    )

    if is_print_test:
        print_genotype_by_strand(
            caller,
            "chr21",
            cur_pos,
            cur_probs_forward,
            cur_probs_reverse,
            args.bayesian_threshold,
            out_fn,
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
    parser.add_argument(
        "--bayesian_threshold",
        type=float,
        help="a threshold that filter genotype with probability, in range of [0,1]",
    )
    args = parser.parse_args()
    predict_bayesian2(args)
    # predict_bayesian1(args)
    # predict_bayesian1_correct_error_within_one_strand(args)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
