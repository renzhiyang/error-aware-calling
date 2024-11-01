import re
import shlex
import argparse

from subprocess import PIPE, Popen


def candidate_information_from(candidate_line):
    """Decode each line of input"""
    candidate = candidate_line.strip().split()
    ctg_name = candidate[0]
    ctg_pos = int(candidate[1])
    ctg_start = ctg_pos
    # ref_base = candidate[2]
    return ctg_name, ctg_start


def samtools_view_from(ctg_name, ctg_start, ctg_end, bam_fn, min_mq, samtools):
    """
    Get reads record by 'samtools view'
    Here region is from candidate_position to candidate_position + 1
    """
    region = "%s:%d-%d" % (ctg_name, ctg_start, ctg_end)
    subprocess = Popen(
        shlex.split(f"{samtools} view -q {min_mq} {bam_fn} {region}"), stdout=PIPE
    )
    return subprocess


def parse_md(md_tag):
    """Parses the MD tag and returns a list of tuples containing matches, mismatches, and deletions."""
    md_parts = re.findall(r"(\d+|\^[A-Za-z]+|[A-Za-z])", md_tag)
    parsed_md = []
    for part in md_parts:
        if part.isdigit():
            parsed_md.append(("M", int(part)))  # Match
        elif part.startswith("^"):
            parsed_md.append(("D", part[1:]))  # Deletion in reference
        else:
            parsed_md.append(("X", part))  # Mismatch
    return parsed_md


def reconstruct_ref_seq(query_seq, md_tag, cigar, start_pos, end_pos):
    """
    Reconstruct the reference and query sequences with gaps ('-') for insertions and deletions
    e.g.,
    query_seq = "GAGAATTTAC"
    md_tag = "2A2^AA2"
    cigar = "2M3I3M2D2M"
    output:
        ref_seq:  GA---ATTAAAC
    """
    ref_seq = ""
    parsed_md = parse_md(md_tag)

    # generate an initial reference sequence
    # In the example, MD: 2A2^AA2. initial ref: MMAMMAAMM
    for tag, part in parsed_md:
        if tag == "M":
            ref_seq += "M" * int(part)
        elif tag in "XD":
            ref_seq += part

    # Then replace "M" and insert "-" to initial ref using CIGAR and query_seq
    length = 0
    query_index = 0
    ref_index = 0
    pointer = start_pos
    final_ref = []
    stop = False
    for b in cigar:
        if stop:
            break

        if b.isdigit():
            length = length * 10 + int(b)
            continue

        if b == "S":
            query_index += length

        elif b in "MX=":
            for _ in range(length):
                if pointer >= end_pos:
                    stop = True
                    break
                query_base = query_seq[query_index]
                ref_base = ref_seq[ref_index]
                if ref_base == "M":
                    final_ref.append(query_base)
                else:
                    final_ref.append(ref_base)
                ref_index += 1
                query_index += 1
                pointer += 1

                # if pointer >= end_pos:  # if reach the candidate position, then return
                #   stop = True
                #   break

        elif b == "I":
            inserted = "-" * length
            final_ref.append(inserted)
            query_index += length

        elif b == "D":
            for _ in range(length):
                if pointer >= end_pos:  # if reach the candidate position, then
                    stop = True
                    break
                delete_base = ref_seq[ref_index]
                final_ref.append(delete_base)
                ref_index += 1
                pointer += 1

                # if pointer >= end_pos:  # if reach the candidate position, then
                #   stop = True
                #   break
        length = 0
    final_ref = "".join(final_ref)

    # print test codes
    """
    if "-" in final_ref:
        print(f'position: {end_pos}')
        #print(f'position: {end_pos}, cigar: {cigar}, md:{md_tag} {parsed_md}, start:{start_pos}')
        print(f'ref: {final_ref[-10:]}')
        #print(f'ini: {ref_seq}')
        print(f'que: {query_seq[-10:]}\n')
    """
    return final_ref


def get_query_base(query_seq, cigar, ref_start, candidate_pos):
    cur_base = ""
    next_ins = "N"
    length = 0
    query_index = 0
    ref_index = ref_start

    for b in cigar:
        if b.isdigit():
            length = length * 10 + int(b)
            continue

        if b == "S":
            query_index += length

        elif b in "MX=":
            for _ in range(length):
                if ref_index <= candidate_pos:
                    cur_base = query_seq[query_index]
                query_index += 1
                ref_index += 1
        elif b == "I":
            if ref_index == candidate_pos + 1:
                next_ins = query_seq[query_index : query_index + length]
            query_index += length
        elif b == "D":
            for _ in range(length):
                if ref_index <= candidate_pos:
                    cur_base = "-"
                ref_index += 1
        length = 0

    if cur_base == "N":  # for some case there are N showed in query reads
        cur_base = "-"
    return cur_base, next_ins


def get_tensor_sequence_from(read, candidate_pos, window_width):
    """
    get the previous 99 ctg bases before current position candidate_pos
    1. for insertion to ctg, add '-' in out ctg bases
    2. 如果ccandidate_pos 位于read的前面几个位置，那么不足99个bases的话，在前面用‘N’补全
    例如 ctg_start = 100， read的start position是80，那么我们首先提取80-99位置的ctg bases
    还有80个base是没有的，那么在这19个base前面添加80个‘N’补齐
    这里采用的是reference sequence作为输入
    """
    read = read.decode()
    read = read.strip().split()
    CAN_POS = candidate_pos  # candidate position, 1-based position in ctg
    QUERY_POS = int(read[3])  # leftmost mapping, 1-based position in ctg
    CIGAR = read[5]
    SEQ = read[9].upper()
    MD = None

    for field in read:
        if field.startswith("MD:Z:"):
            MD = field[5:]
            break

    if MD is None:
        print()

    # get the previous reference bases before the current candidate position
    ref_seq = reconstruct_ref_seq(SEQ, MD, CIGAR, QUERY_POS, CAN_POS)
    cur_base, next_ins = get_query_base(SEQ, CIGAR, QUERY_POS, CAN_POS)

    if window_width > len(ref_seq):
        padding_seq = (window_width - len(ref_seq)) * "N"  # 'N' represent padding base
        ref_seq = padding_seq + ref_seq
    else:
        ref_seq = ref_seq[len(ref_seq) - window_width :]

        # if next_ins != "N":
    # if cur_base == "-":
    #     print(
    #         f"start_pos: {QUERY_POS}, end_pos: {CAN_POS}, cur_base: {cur_base}, next_ins: {next_ins}"
    #     )
    #     print(f"ref_seq: {ref_seq}")
    return ref_seq, cur_base, next_ins


def create_tensor(args):
    samtools = args.samtools
    # ref_fn = args.ref_fn
    bam_fn = args.bam_fn
    candidates_fn = args.candidates_fn
    tensor_fn = args.tensor_fn
    min_mq = args.min_mq
    window_width = int(args.tensor_window_width)

    candidates = open(candidates_fn, "r")
    output_tensor_fn = open(tensor_fn, "w")
    for line in candidates:
        ctg_name, candidate_pos = candidate_information_from(line)
        subprocess = samtools_view_from(
            ctg_name, candidate_pos, candidate_pos + 1, bam_fn, min_mq, samtools
        )
        reads = subprocess.stdout  # type: ignore

        if reads is None:
            continue

        for i, read in enumerate(reads):
            tensor_sequence, cur_base, next_ins = get_tensor_sequence_from(
                read, candidate_pos, window_width
            )
            if cur_base == "":  # for some reads, there are all ""
                continue

            if len(next_ins) in range(3, 7):  # exclude insertion length large than 6
                next_ins = "rep" + str(len(next_ins))
            elif len(next_ins) > 6:
                continue

            output_line = (
                f"{ctg_name}"
                + "\t"
                + f"pos_{candidate_pos}_{i}"
                + "\t"
                + tensor_sequence
                + "\t"
                + cur_base
                + "\t"
                + next_ins
                + "\n"
            )
            output_tensor_fn.write(output_line)


def main():
    parser = argparse.ArgumentParser(
        description="Generate tensor around candidates from alignment file for trained model"
    )
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
        "--candidates_fn",
        type=str,
        help="Path to candidate sites generated by find_candidates.py",
    )
    parser.add_argument(
        "--tensor_fn", type=str, help="Tensor output, will be used to call variants"
    )
    parser.add_argument(
        "--min_mq", type=int, default=20, help="Minimum mapping quality of a read"
    )
    parser.add_argument(
        "--tensor_window_width",
        type=int,
        default=99,
        help="The width of tensor window, for illumina reads default is 99",
    )
    args = parser.parse_args()

    create_tensor(args)


if __name__ == "__main__":
    main()
