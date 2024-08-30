import math
import argparse
import numpy as np
import src.utils as utils

from src.bayesian import BayesianCaller

VOCAB = {"A": 0, "C": 1, "G": 2, "T": 3, "-": 4}


def main(args):
    CANDIDATES = open(args.candidates_fn, "r")
    post_probs_in_pos = {}

    for line in CANDIDATES:
        line = line.strip().split()
        CTG = line[0]
        POSITION = int(line[1])
        REF = line[2]
        DEPTH = int(line[3])
        ALLELE_1 = line[4] if line[4] != "D" else "-"
        ALLELE_1_COUNT = int(line[5])
        ALLELE_2 = line[6] if line[6] != "D" else "-"
        ALLELE_2_COUNT = int(line[7])

        # TODO: Insertion
        # skip insertion now, we should process original reads
        # if we wan to process insertion
        if ALLELE_1 in ("I", "N") or ALLELE_2 in ("I", "N"):
            continue

        caller = BayesianCaller()

        # Initialize cur_base_dis and ins_base_dis
        # They are used to simulate the outputs of neural network
        cur_base_dis = np.ones(5, dtype=np.float32) * 1e-4
        ins_base_dis = np.ones(25, dtype=np.float32) * 1e-4
        cur_base_dis[VOCAB[ALLELE_1]] = ALLELE_1_COUNT / DEPTH
        cur_base_dis[VOCAB[ALLELE_2]] = ALLELE_2_COUNT / DEPTH
        print(f"position: {POSITION}, cur_base_dis: {cur_base_dis}")

        # start calling by BayesianCaller
        if POSITION not in post_probs_in_pos:
            post_probs_in_pos[POSITION] = []

        for _ in range(ALLELE_1_COUNT):
            genotypes_one_read = caller.all_genotypes_posterior_prob_per_read_log(
                cur_base_dis, ins_base_dis, ALLELE_1, "N"
            )

            if len(post_probs_in_pos[POSITION]) == 0:
                post_probs_in_pos[POSITION] = genotypes_one_read
            else:
                post_probs_in_pos[POSITION] = caller.add_pos_probs_of_two_reads(
                    post_probs_in_pos[POSITION], genotypes_one_read
                )

        for _ in range(ALLELE_2_COUNT):
            genotypes_one_read = caller.all_genotypes_posterior_prob_per_read_log(
                cur_base_dis, ins_base_dis, ALLELE_2, "N"
            )

            if len(post_probs_in_pos[POSITION]) == 0:
                post_probs_in_pos[POSITION] = genotypes_one_read
            else:
                post_probs_in_pos[POSITION] = caller.add_pos_probs_of_two_reads(
                    post_probs_in_pos[POSITION], genotypes_one_read
                )
        # normalize the posterior probs by dividing marginal likelihoods
        # post_probs_in_pos[POSITION] = caller.normalize_pos_probs_from_minus_input(
        #    post_probs_in_pos[POSITION]
        # )

        #### check if the most probable genotype is homozygous reference
        sorted_dic = sorted(
            post_probs_in_pos[POSITION].items(), key=lambda x: x[1], reverse=True
        )
        sorted_dic = sorted_dic[:2]
        # normalize the first two genotypes, noticed that probs are minus log probs now
        marginal_likelihood = sum([1 / abs(value) for key, value in sorted_dic])
        for i in range(len(sorted_dic)):
            key, value = sorted_dic[i]
            sorted_dic[i] = (key, (1 / abs(value)) / marginal_likelihood)

        genotype_1 = sorted_dic[0][0]
        genotype_1_type = genotype_1.split("_")[0]
        allele1 = genotype_1.split("_")[1]
        allele2 = genotype_1.split("_")[2]
        if genotype_1_type == "snv" and REF == allele1 and REF == allele2:
            continue
        # quality = -10 * math.log10(sorted_dic[1][1] / sorted_dic[0][1])
        quality = -10 * math.log10(sorted_dic[1][1])
        print(
            f"position: {POSITION}, post_probs: {sorted_dic}, QUAL: {quality}, marginal: {sum(post_probs_in_pos[POSITION].values())} \n"
        )
        if quality > 20:
            print(f"position: {POSITION}, post_probs: {sorted_dic}, QUAL: {quality} \n")

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates_fn",
        type=str,
        default="data/candidates.txt",
        help="candidate sites file",
    )
    args = parser.parse_args()
    main(args)
