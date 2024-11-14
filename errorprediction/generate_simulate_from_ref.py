import os
import argparse
import errorprediction.utils as utils


def print_label(args, up_seq, ref_pos):
    read_base = up_seq[-1]
    for i in range(args.depth):
        utils.print_label(
            chrom=args.ctg_name,
            type="Positive",
            read_strand="forward",
            position=ref_pos,
            label=read_base,
            read_base=read_base,
            ref_base=read_base,
            alts="",
            is_germline="No",
            variant_type="SNV",
            sequence_around=up_seq,
            file_p=args.out_fn,
        )


def generate_simulate_data(args):
    subprocess = utils.samtools_faidx_from(
        ctg_name=args.ctg_name,
        ctg_start=args.ctg_start,
        ctg_end=args.ctg_end,
        fasta_fn=args.ref_fn,
        samtools=args.samtools,
    )
    reference_seq = subprocess.stdout
    if reference_seq is None:
        return

    window_size = args.up_len + 1
    window = []
    ref_pos = args.ctg_start
    for _, seq in enumerate(reference_seq):
        seq = seq.decode("utf-8").strip()
        if seq.startswith(">"):
            continue

        for char in seq:
            if char == "N":
                ref_pos += 1
                continue

            window.append(char)
            if len(window) == window_size:
                up_seq = "".join(window)
                print_label(args, up_seq, ref_pos)
                window.pop(0)
                ref_pos += 1


def main():
    parser = argparse.ArgumentParser(
        description="generate simulation data only from reference genome"
    )
    parser.add_argument(
        "--ref_fn",
        type=str,
        help="reference genome fasta file",
        default="",
        required=True,
    )
    parser.add_argument(
        "--out_fn", type=str, help="output file", default="./simulate.label"
    )
    parser.add_argument(
        "--ctg_name", type=str, help="contig name", default="", required=True
    )
    parser.add_argument(
        "--ctg_start", type=int, help="contig start", default=0, required=True
    )
    parser.add_argument(
        "--ctg_end", type=int, help="contig end", default=0, required=True
    )
    parser.add_argument(
        "--up_len",
        type=int,
        default=99,
        help="the length of upstream sequence also the input window length of NN, default is 99",
    )
    parser.add_argument("--depth", type=int, default=30, help="simulate depth")
    parser.add_argument(
        "--is_padding_N",
        type=bool,
        default=False,
        help="whether padding N in the start of upsequence",
    )
    parser.add_argument(
        "--samtools", type=str, default="samtools", help="samtools path"
    )
    args = parser.parse_args()
    generate_simulate_data(args)


if __name__ == "__main__":
    main()
