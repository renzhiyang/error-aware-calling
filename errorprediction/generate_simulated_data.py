import os
import argparse
import errorprediction.utils as utils


class Simulater:
    def __init__(
        self,
        bam_file: str,
        out_file: str,
        samtools: str,
        ctg_name: str,
        ctg_start: int,
        ctg_end: int,
        up_len: int = 99,
        num_samples: int = 10000,
        is_padding_N: bool = False,
        random_ratio: float = 0.05,
    ):
        self.bam_file = bam_file
        self.out_file = out_file
        self.samtools = samtools
        self.up_len = up_len
        self.real_num = int(num_samples * (1 - random_ratio))
        self.random_num = num_samples - self.real_num
        self.ctg_name = ctg_name
        self.ctg_start = ctg_start
        self.ctg_end = ctg_end
        self.is_padding_N = is_padding_N

        dir = os.path.dirname(out_file)
        if not os.path.exists(dir):
            print(f"Creating directory for {out_file}")
            os.makedirs(dir)

    def generate_data(self):
        subprocess = utils.samtools_view_from(
            self.ctg_name,
            self.ctg_start,
            self.ctg_end,
            self.bam_file,
            20,
            self.samtools,
        )
        reads = subprocess.stdout

        if reads is None:
            print("There are no reads in this region")
            return None

        for i, read in enumerate(reads):
            read = read.decode("utf-8").strip()
            if read.startswith("@"):
                continue
            self.__print_data_from_read(read)

    def __print_seq(self, query_pos, query_seq):
        if self.is_padding_N:
            query_seq = "N" * (self.up_len // 2) + query_seq

        len_seq = len(query_seq)
        if len_seq <= self.up_len + 1:
            return

        sequence_around = ""
        end_pointer = len_seq - self.up_len
        for i in range(end_pointer + 1):
            pos = query_pos + i
            sequence_around = query_seq[i : i + self.up_len + 1]
            read_base = sequence_around[-1]
            ref_base = sequence_around[-1]
            utils.print_label(
                chrom="simulate",
                type="Positive",
                read_strand="forward",
                position=pos,
                label=ref_base,
                read_base=read_base,
                ref_base=ref_base,
                alts="",
                is_germline="Yes",
                variant_type="SNV",
                sequence_around=sequence_around,
                file_p=self.out_file,
            )
            # only print one for each sequence now
            break

    def __print_data_from_read(self, read):
        read = read.split("\t")
        FLAG = int(read[1])
        QUERY_POS = int(read[3])
        CIGAR = read[5]
        SEQ = read[9].upper()
        STRAND = "forward" if FLAG & 16 == 0 else "reverse"
        MD = ""

        # Get the MD tag
        for field in read:
            if field.startswith("MD:Z:"):
                MD = field[5:]
                break
        if not MD.isdigit():
            return

        # Get the original sequence if the read is mapped to the reverse strand
        if STRAND == "reverse":
            SEQ = utils.reverse_complement(SEQ)

        # Initialize the upstream sequence
        # if we want to padding N in the start of upseq, the start length is up_len / 2,
        # up_seq = "N" * (self.up_len // 2) if self.is_padding_N else ""

        length = 0
        query_index = 0
        query_pos = QUERY_POS
        for b in CIGAR:
            if b.isdigit():
                length = length * 10 + int(b)
                continue

            if b in "SID":
                return

            elif b in "MX=":
                query_seq = SEQ[query_index : query_index + length]
                self.__print_seq(query_pos, query_seq)


def generate_simulate_data(args):
    simulater = Simulater(
        args.bam_fn,
        args.out_fn,
        args.samtools,
        args.ctg_name,
        args.ctg_start,
        args.ctg_end,
        args.up_len,
        args.num_samples,
        args.is_padding_N,
    )
    simulater.generate_data()


def main():
    parser = argparse.ArgumentParser(description="generate simulation data for test")
    parser.add_argument(
        "--bam_fn", type=str, help="input alignment bam file", default="", required=True
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
    parser.add_argument(
        "--num_samples", type=int, default=100, help="number of simulated samples"
    )
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
