import argparse
import numpy as np

from src.bayesian import BayesianCaller

VOCAB = {"A": 0, "C": 1, "G": 2, "T": 3, "-": 4}
INS = [
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
VOCAB_INS = {item: index for index, item in enumerate(INS)}


def test(args):
    caller = BayesianCaller()
    candidate_fn = args.candidates_fn
    candidates = open(candidate_fn, "r").readlines()
    pos_probs_in_pos = {}

    count = 0
    for candidate in candidates:
        candidate = candidate.strip().split()
        ctg_name = candidate[0]
        position = int(candidate[1])
        ref_b = candidate[2]
        depth = int(candidate[3])
        top1_base = candidate[4]
        top1_num = int(candidate[5])
        top2_base = candidate[6]
        top2_num = int(candidate[7])
        ins_base = candidate[12]
        ins_num = int(candidate[13])
        del_base = candidate[14]
        del_num = int(candidate[15])

        cur_base_dis = [1e-4, 1e-4, 1e-4, 1e-4, 1e-4]
        ins_base_dis = np.ones(25, dtype=np.float32)
        ins_base_dis *= 0.9
        cur_base_dis[VOCAB[top1_base]] = top1_num / depth
        cur_base_dis[VOCAB[top2_base]] = top2_num / depth
        # print(f'position: {position}, cur_base_dis: {cur_base_dis}"')

        print(f'position: {position}, cur_base_dis: {cur_base_dis}"')

        if position not in pos_probs_in_pos:
            pos_probs_in_pos[position] = []

        for _ in range(top1_num):
            # print(f"position: {position}, top1_base: {top1_base}")
            genotypes_one_read = caller.all_genotypes_posterior_prob_per_read_2_log(
                cur_base_dis, ins_base_dis, top1_base, "N"
            )

            if len(pos_probs_in_pos[position]) == 0:
                pos_probs_in_pos[position] = genotypes_one_read
            else:
                # print(pos_probs_in_pos[position])
                pos_probs_in_pos[position] = caller.add_pos_probs_of_two_reads(
                    pos_probs_in_pos[position], genotypes_one_read
                )

        # print(f"position: {position}, pos_probs: {pos_probs_in_pos[position]}")

        for _ in range(top2_num):
            # print(f"position: {position}, top2_base: {top2_base}")
            genotypes_one_read = caller.all_genotypes_posterior_prob_per_read_2_log(
                cur_base_dis, ins_base_dis, top2_base, "N"
            )
            # print(
            #    f"genotypes_one_read:{genotypes_one_read}, pos_probs_in_pos: {pos_probs_in_pos[position]}"
            # )
            # pos_probs_in_pos[position] = caller.add_pos_probs_of_two_reads(
            #    pos_probs_in_pos[position], genotypes_one_read
            # )

            if len(pos_probs_in_pos[position]) == 0:
                pos_probs_in_pos[position] = genotypes_one_read
            else:
                pos_probs_in_pos[position] = caller.add_pos_probs_of_two_reads(
                    pos_probs_in_pos[position], genotypes_one_read
                )
        sorted_dic = dict(
            sorted(pos_probs_in_pos[position].items(), key=lambda x: x[1], reverse=True)
        )

        print(
            f"position: {position}, depth: {top1_num + top2_num}, pos_probs: {sorted_dic}\n"
        )

        # test codes
        count += 1
        if count >= 20:
            break


def test_snv_diploid_threshold(depth: int, first_allele_ratio: float):
    caller = BayesianCaller()
    pos_probs_in_pos = {}

    top1_base = "A"
    top2_base = "G"
    second_allele_ratio = 1 - first_allele_ratio
    cur_base_dis = [1e-4, 1e-4, 1e-4, 1e-4, 1e-4]

    top1_num = int(depth * first_allele_ratio)
    top2_num = int(depth * second_allele_ratio)
    ins_base_dis = np.ones(25, dtype=np.float32) * 1e-4
    cur_base_dis[VOCAB[top1_base]] = first_allele_ratio
    cur_base_dis[VOCAB[top2_base]] = second_allele_ratio

    for _ in range(top1_num):
        genotypes_one_read = caller.all_genotypes_posterior_prob_per_read_2_log(
            cur_base_dis, ins_base_dis, top1_base, "N"
        )

        if len(pos_probs_in_pos) == 0:
            pos_probs_in_pos = genotypes_one_read
        else:
            pos_probs_in_pos = caller.add_pos_probs_of_two_reads(
                pos_probs_in_pos, genotypes_one_read
            )

    for _ in range(top2_num):
        genotypes_one_read = caller.all_genotypes_posterior_prob_per_read_2_log(
            cur_base_dis, ins_base_dis, top2_base, "N"
        )
        pos_probs_in_pos = caller.add_pos_probs_of_two_reads(
            pos_probs_in_pos, genotypes_one_read
        )

    sorted_genotypes = sorted(
        pos_probs_in_pos.items(), key=lambda x: x[1], reverse=True
    )
    print(f"number of genotypes: {len(sorted_genotypes)}")
    print(
        f"depth: {depth}, first_allele_ratio: {first_allele_ratio}, sorted_genotypes: {sorted_genotypes}"
    )


def test_snv_diploid_range():
    # depths = range(30, 101, 10)
    depths = [30]
    first_allele_ratios = np.linspace(0.9, 0.5, 5)

    for depth in depths:
        for ratio in first_allele_ratios:
            print(f"test depth: {depth}, first allele ratio: {ratio}")
            test_snv_diploid_threshold(depth, ratio)
            print("\n")


def test_ins_diploid_threshold(depth: int, first_allele_ratio: float):
    caller = BayesianCaller()
    post_probs_in_pos = {}
    observed_base = "A"
    top1_ins = "A"
    top2_ins = "AA"
    second_allele_ratio = 1 - first_allele_ratio
    cur_base_dis = [1e-4, 1e-4, 1e-4, 1e-4, 0.8]

    top1_num = int(depth * first_allele_ratio)
    top2_num = int(depth * second_allele_ratio)
    ins_base_dis = np.ones(25, dtype=np.float32) * 1e-4
    ins_base_dis[VOCAB_INS[top1_ins]] = first_allele_ratio
    ins_base_dis[VOCAB_INS[top2_ins]] = second_allele_ratio

    for _ in range(top1_num):
        genotypes_one_read = caller.all_genotypes_posterior_prob_per_read_2_log(
            cur_base_dis, ins_base_dis, observed_base, top1_ins
        )
        if len(post_probs_in_pos) == 0:
            post_probs_in_pos = genotypes_one_read
        else:
            post_probs_in_pos = caller.add_pos_probs_of_two_reads(
                post_probs_in_pos, genotypes_one_read
            )

    for _ in range(top2_num):
        genotypes_one_read = caller.all_genotypes_posterior_prob_per_read_2_log(
            cur_base_dis, ins_base_dis, observed_base, top2_ins
        )
        post_probs_in_pos = caller.add_pos_probs_of_two_reads(
            post_probs_in_pos, genotypes_one_read
        )

    sorted_genotypes = sorted(
        post_probs_in_pos.items(), key=lambda x: x[1], reverse=True
    )
    print(f"number of genotypes: {len(sorted_genotypes)}")
    print(
        f"depth: {depth}, first_allele_ratio: {first_allele_ratio}, sorted_genotypes: {sorted_genotypes[:10]}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="call variants using fine-turned model"
    )
    parser.add_argument(
        "--candidates_fn",
        type=str,
        help="candidate file",
        required=True,
    )
    # args = parser.parse_args()
    # test(args)
    # test_snv_diploid_threshold(100, 0.82)  # 0.83->0.82 from haploid to diploid
    # test_snv_diploid_range()
    test_ins_diploid_threshold(100, 0.5)


if __name__ == "__main__":
    main()
