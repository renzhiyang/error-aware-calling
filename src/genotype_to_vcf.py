import re
import os
import argparse


def load_ref_seq(ref_fn, ctg_name):
    ref_f = open(ref_fn, "r")
    is_ctg = False
    ref_seq = ""
    for line in ref_f:
        line = line.strip()

        if line.startswith(">"):
            if ctg_name in line:
                is_ctg = True
            else:
                is_ctg = False
        elif is_ctg:
            ref_seq += line
    return ref_seq


def write_header(vcf_f, ref_fn, ctg_name, len_ref):
    header = (
        "##fileformat=VCFv4.2\n"
        f"##source=custom_script\n"
        f"##reference={ref_fn}\n"
        f"##contig=<ID={ctg_name},length={len_ref},assembly=GRCh38,"
        "md5=b18e6c531b0bd70e949a7fc20859cb01,species='Homo sapiens',taxonomy=x>\n"
        "##FORMAT=<ID=GT,Number=1,Type=String,Description='Genotype'>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n"
    )
    vcf_f.write(header)


def out_vcf_snv(
    ctg_name, position, ref_base, snv_allele_1, snv_allele_2, snv_value, vcf_f
):
    if snv_allele_1 == "-" or snv_allele_2 == "-":
        # TODO currently only consider snv, exclude deletions
        return

    # exclude homozygous reference
    if snv_allele_1 == ref_base and snv_allele_2 == ref_base:
        return 

    if snv_allele_1 == ref_base:
        ref = snv_allele_1
        alt = snv_allele_2
        genotype = "0/1"
    elif snv_allele_2 == ref_base:
        ref = snv_allele_2
        alt = snv_allele_1
        genotype = "0/1"
    else:
        ref = ref_base
        if snv_allele_1 == snv_allele_2:
            alt = snv_allele_1
            genotype = "1/1"
        else:
            alt = f"{snv_allele_1},{snv_allele_2}"
            genotype = "1/2"

    vcf_line = f"{ctg_name}\t{position}\t.\t{ref}\t{alt}\t{snv_value}\tPASS\t.\tGT\t{genotype}\n"
    vcf_f.write(vcf_line)
    # print(vcf_line)


def convert_genotype_to_vcf(args):
    GENO_FN = args.geno_fn
    REF_FN = args.ref_fn
    VCF_FN = args.vcf_fn
    CTG_NAME = args.ctg_name

    geno_f = open(GENO_FN, "r")
    out_vcf = open(VCF_FN, "w")

    position_pattern = r"position:\s*(\d+)"
    snv_pattern = r"snv:\s*\[\('([^']+)',\s*([-+]?\d*\.\d+|\d+)\)"
    insertion_pattern = r"ins:\s*\[\('([^']+)',\s*([-+]?\d*\.\d+|\d+)\)"

    ref_seq = load_ref_seq(REF_FN, CTG_NAME)
    len_ref_seq = len(ref_seq)

    # write header info to vcf file
    write_header(out_vcf, REF_FN, CTG_NAME, len_ref_seq)

    for line in geno_f:
        position_match = re.search(position_pattern, line)
        position = position_match.group(1) if position_match else None
        position = int(position) if position is not None else None

        snv_match = re.search(snv_pattern, line)
        snv_type, snv_value = snv_match.groups() if snv_match else (None, None)

        ins_match = re.search(insertion_pattern, line)
        ins_type, ins_value = ins_match.groups() if ins_match else (None, None)

        if snv_type is None or ins_type is None or position is None:
            continue

        ref_base = ref_seq[position - 1]

        # print(
        #    f"{position} ref:{ref_seq[position-1]} {snv_type}:{snv_value}, {ins_type}:{ins_value}"
        # )
        snv_type = snv_type.split("_")
        snv_allele_1, snv_allele_2 = snv_type[1], snv_type[2]
        out_vcf_snv(
            CTG_NAME, position, ref_base, snv_allele_1, snv_allele_2, snv_value, out_vcf
        )


def main():
    parser = argparse.ArgumentParser(
        description="convert output genotype file to vcf file"
    )
    parser.add_argument(
        "--ref_fn",
        type=str,
        help="the reference file that used to call variants",
        required=True,
    )
    parser.add_argument(
        "--geno_fn",
        type=str,
        help="the output genotype file produced by predict.py",
        required=True,
    )
    parser.add_argument("--ctg_name", type=str, help="chromosome name", required=True)
    parser.add_argument("--vcf_fn", type=str, help="the output vcf file", required=True)
    args = parser.parse_args()
    convert_genotype_to_vcf(args)


if __name__ == "__main__":
    main()
