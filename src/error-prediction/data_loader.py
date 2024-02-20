import pysam
import hydra
import gzip

from omegaconf import DictConfig, OmegaConf
from typing import List, Dict


# Load variants from　vcf file
# Return is a  two-level dictionary
# key-1: chromosome, key-2: position
# value: reference base, alternates base, is_indel
def read_vcf(filename: str) -> dict:
    variants = {}
    open_func = gzip.open if filename.endswith('.gz') else open
    with open_func(filename, 'rt') as file:
        for line in file:
            if line.startswith('#'):
                continue # skip header lines
            columns = line.strip().split("\t")
            chr = columns[0]
            pos = int(columns[1])
            ref = columns[3]
            alts = columns[4].split(',')
            is_indel = len(ref) > 1 or any(len(alt)>1 for alt in alts)
            
            if chr not in variants:
                variants[chr] = {}
            variants[chr][pos] = {'ref': ref, 'alts': alts, 'is_indel': is_indel}
    return variants


# Load confident regions
# return confident regions
# key: chromosome name. Value: start position, end position
def read_bed(filename: str):
    regions = {}
    open_func = gzip.open if filename.endswith('.gz') else open
    with open_func(filename, 'rt') as file:
        for line in file:
            if line.startswith('#') or line.strip() == '':
                continue
            columns = line.strip().split('\t')
            chr, start_pos, end_pos = columns[0], int(columns[1]), int(columns[2])
            if chr not in regions:
                regions[chr] = []
            regions[chr].append((start_pos, end_pos))
    return regions
            

# The function to identify if the current base in sequencing read support ref/variant or not.
# If the current base not support ref/variant, it will be recongnized as sequencing error.
# It only work in highly condident regions, errors are used to build training data
def is_error(position, base, is_indel):
    return None


# To verify if read in confident regions
def is_read_in_confident_region(read: pysam.AlignedSegment, confident_regions: list) -> bool:
    for start, end in confident_regions:
        if read.reference_start < end and read.reference_end > start:
            return True
    return False


# Output labeled data to a file
def print_labeled_data(chrom: str, read: pysam.AlignedSegment, variants: dict):
    # TODO: iterate all bases in read, if the base support reference or variants, it is marked as
    # positive label. Otherwise it is marked as negative label.
    # output these information each line:
    # chrom, position, positive/negative label, read base, reference/variant base, sequence(200 bases before and after the current locus)  
    return


def generate_label(cfg):
    bam_file = pysam.AlignmentFile(cfg.data_path.bam_f)
    confident_regions = read_bed(cfg.data_path.confident_f) 
    variants_dict = read_vcf(cfg.data_path.vcf_f)
    for read in bam_file:
        if read.mapping_quality <= cfg.alignment_filter.min_mapping_quality:
            continue # Filter reads with lower mapping quality
        
        chrom = bam_file.get_reference_name(read.reference_id)
        if chrom in confident_regions and is_read_in_confident_region(read, confident_regions[chrom]):
            print_labeled_data(chrom, read, variants_dict[chrom])
    

@hydra.main(version_base=None, config_path='../../configs/error-prediction', config_name='params.yaml')
def main(cfg: DictConfig) -> None:
    generate_label(cfg) 


if __name__ == '__main__':
    main()
    
            