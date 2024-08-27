# Extract candiate variants from BAM file
# python find_candidates.py \
#    --ref_fn /home/yang1031/genomeData/ONT_open_datasets/giab_2023.05/analysis/benchmarking/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna \
#    --bam_fn /home/yang1031/genomeData/Illumina/hg002/precisionFDA/subset/HG002.novaseq.pcr-free-grch38-tagged-chr1.35x.bam \
#    --ctg_name chr1 --ctg_start 826400 --ctg_end 827000 --min_mq 20 --min_coverage 10 --min_allele_freq 0.125
# output example:
#   ctg_name, pos, ref_base, depth, base-num
#   chr1 826420 A 12 T 11 A 1 C 0 G 0 I 0 D 0 N 0
#   chr1 826431 G 13 A 12 G 1 C 0 T 0 I 0 D 0 N 0
#   chr1 826577 A 19 A 19 I 19 C 0 G 0 T 0 D 0 N 0
#   chr1 826893 G 47 A 45 G 2 C 0 T 0 I 0 D 0 N 0

import sys
import shlex
import utils
import argparse

from subprocess import PIPE, Popen
from collections import defaultdict


def reference_sequence_from(samtools, fasta, regions):
    region_str = " ".join(regions)
    with Popen(shlex.split(f"{samtools} faidx {fasta} {region_str}"), stdout=PIPE) as p:
        lines = p.stdout.read().decode().splitlines()  # type: ignore
    return "".join(lines[1:]).upper()


def exclude_many_clipped_bases_read(CIGAR):
    clipped_length = 0
    total_length = 0
    length = 0
    for b in CIGAR:
        if b.isdigit():
            length = length * 10 + int(b)
            continue

        if b == "S":
            clipped_length += length
        total_length += length
        length = 0

    return 1.0 - float(clipped_length) / (total_length + 1) < 0.55


def preprocess_read(row, ctg_name, minimum_mapping_quality, pileup):
    columns = row.strip().split()
    if (
        columns[0][0] == "@"  # exlude head line
        or columns[2] != ctg_name  # exclude reads aligned to other ctg
        or int(columns[4])
        < minimum_mapping_quality  # exlude reads with lower mapping quality
    ):
        return

    # exlude reads with many soft clipped bases
    CIGAR = columns[5]
    if exclude_many_clipped_bases_read(CIGAR):
        return

    POS = int(columns[3]) - 1
    SEQ = columns[9].upper()
    reference_position = POS
    query_position = 0
    length = 0
    for b in CIGAR:
        if b.isdigit():
            length = length * 10 + int(b)
            continue

        if b == "S":
            query_position += length

        elif b in "MX=":
            for _ in range(length):
                base = utils.process_base(SEQ[query_position])
                pileup[reference_position][base] += 1
                reference_position += 1
                query_position += 1

        elif b == "I":
            pileup[reference_position - 1]["I"] += 1
            query_position += length

        elif b == "D":
            for _ in range(length):
                # print(f"position: {reference_position}is deletion")
                pileup[reference_position]["D"] += 1
                reference_position += 1

        length = 0


def get_candidates(args):
    samtools = args.samtools
    fasta_file = args.ref_fn
    regions = (
        [f"{args.ctg_name}:{args.ctg_start}-{args.ctg_end}"] if args.ctg_name else []
    )
    pileup = defaultdict(
        lambda: {"A": 0, "C": 0, "G": 0, "T": 0, "I": 0, "D": 0, "N": 0}
    )

    reference_sequence = reference_sequence_from(samtools, fasta_file, regions)
    if reference_sequence is None or len(reference_sequence) == 0:
        print("[ERROR] Failed to load reference sequence.", file=sys.stderr)
        sys.exit(1)

    counts = 0
    with Popen(
        # shlex.split(
        # f"{samtools} view {args.bam_fn} {args.ctg_name} {' '.join(regions)}"
        # ),
        shlex.split(f"{samtools} view {args.bam_fn} {''.join(regions)}"),
        stdout=PIPE,
    ) as p:
        for row in p.stdout:  # type: ignore
            # counts += 1
            # if counts % 10000 == 0:
            # print(counts, flush=True)
            preprocess_read(row.decode(), args.ctg_name, args.min_mq, pileup)

    positions = sorted(pileup.keys())
    for zero_based_position in positions:
        if args.ctg_start <= zero_based_position + 1 <= args.ctg_end:
            # print test

            reference_position = zero_based_position - args.ctg_start + 1
            ref_base = reference_sequence[reference_position]
            base_count = list(pileup[zero_based_position].items())
            depth = (
                sum(x[1] for x in base_count)
                # - pileup[zero_based_position]["I"]
                # - pileup[zero_based_position]["D"]
            )
            if depth < args.min_coverage:
                continue
            base_count.sort(key=lambda x: -x[1])
            # print(f"position: {zero_based_position + 1}, base_count:{base_count}")

            # print candidate positions
            is_first_allele_fits = base_count[0][0] != ref_base
            is_second_allele_fits = (
                base_count[0][0] == ref_base
                and (float(base_count[1][1]) / depth) >= args.min_allele_freq
            )
            if is_first_allele_fits or is_second_allele_fits:
                print(
                    f"{args.ctg_name} {zero_based_position + 1} {ref_base} {depth} "
                    + " ".join([f"{x[0]} {x[1]}" for x in base_count]),
                    flush=True,
                )

        del pileup[zero_based_position]


def main():
    parser = argparse.ArgumentParser(description="find candidtate variants")
    parser.add_argument(
        "--ref_fn", type=str, help="reference file name", default="", required=True
    )
    parser.add_argument(
        "--bam_fn", type=str, help="input alignment bam file", default="", required=True
    )
    parser.add_argument(
        "--samtools", type=str, help="Path to the samtools", default="samtools"
    )
    parser.add_argument(
        "--min_coverage",
        type=int,
        help="minimum coverage for find candidates, default is 10",
        default=10,
    )
    parser.add_argument(
        "--min_allele_freq",
        type=float,
        help="Minimum allele frequency for find candidates, default is 0.125",
        default=0.125,
    )
    parser.add_argument(
        "--min_mq",
        type=int,
        help="minimum mapping quality required, default is 10",
        default=10,
    )
    parser.add_argument(
        "--min_bq",
        type=int,
        help="minimum base quality required, default is 10",
        default=10,
    )
    parser.add_argument("--vcf_fn", type=str, help="candidate sites vcf file")
    parser.add_argument(
        "--ctg_name",
        type=str,
        help="the name of suquence name, e.g., chr1",
        required=True,
    )
    parser.add_argument(
        "--ctg_start", type=int, help="start position of sequence", required=True
    )
    parser.add_argument(
        "--ctg_end", type=int, help="end position of sequence", required=True
    )
    args = parser.parse_args()

    get_candidates(args)


if __name__ == "__main__":
    main()
