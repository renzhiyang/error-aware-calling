import pysam
import hydra
import gzip

from omegaconf import DictConfig, OmegaConf
from typing import List, Dict


# Load variants from　vcf file
# Return is a  two-level dictionary
# key-1: chromosome, key-2: position
# value: reference base, alternates base, is_indel
def load_vcf(filename: str) -> dict:
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
def load_bed(filename: str):
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


def load_ref(filename: str):
    reference = {}
    with pysam.FastxFile(filename) as fh:
        for entry in fh:
            reference[entry.name] = entry.sequence
    return reference


# To verify if read in confident regions
def is_read_in_confident_region(read: pysam.AlignedSegment, confident_regions: list) -> bool:
    for start, end in confident_regions:
        if read.reference_start < end and read.reference_end > start:
            return True
    return False

"""
Extract the sequence around a given position with a specified window size.
"""
def get_sequence_around(full_read_sequence:str, position:int, config:DictConfig):
    window_half = config.label.window_size_half
    start = max(0, position - window_half)
    end = min(len(full_read_sequence), position + window_half + 1)
    return full_read_sequence[start:end]


def print_label(chrom:str, position:int, label:str, 
                read_base:str, ref_base:str, alts:str,
                variant_type:str, sequence_around:str, config:DictConfig):
    label_f = open(config.data_path.label_f, "a")
    print(
    f'chrom:{chrom} '
    f'pos_ref:{position} '
    f'label:{label} '
    f'read_base:{read_base} '
    f'ref_base:{ref_base} '
    f'alts:{alts} '
    f'variant_type:{variant_type} '
    f'sequence_around:{sequence_around}',
    flush=True, file=label_f
)


""" 
For Indels.
"""
def handle_indel_loci(chrom:str, indel_type:str, indel_sequence:str, ref_base:str, 
                      start_position:int, variants:dict,
                      full_read_sequence:str, config:DictConfig):
    # Assuming indel_type is either "insertion" or "deletion"
    variant_info = variants.get(start_position, {'ref': None, 'alts': {}})
    variant_ref = variant_info.get('ref', [])
    variant_alts = variant_info.get('alts', [])
    
    # TODO: add codes to process indels, now don't consider the indels in non-variant regions
    if start_position == None or variant_ref == None or variant_alts == None:
        return
    
    #print(chrom, start_position, ref_base, indel_sequence, variant_ref, variant_alts)
    label = 'positive' if indel_sequence in variant_ref or indel_sequence in variant_alts else "negative"
    # For deletions, the sequence around is based on the reference start position
    sequence_around = get_sequence_around(full_read_sequence, start_position, config)
    print(sequence_around, start_position, len(full_read_sequence))
    print_label(chrom, start_position, label, indel_sequence, 
                ref_base, f'{variant_ref} {variant_alts}', 
                indel_type, sequence_around, config)


"""
For SNPs
The function to identify if the current base in sequencing read support ref/variant or not.
If the current base not support ref/variant, it will be recongnized as sequencing error.
It only work in highly condident regions, errors are used to build training data
"""
def handle_snp_loci(chrom:str, ref_pos:int, read_base:str, ref_base:str, variants:dict,
               full_read_sequence:str, read_pos:int, config:DictConfig):
    variant_info = variants.get(ref_pos, {'ref': None, 'alts': {}})
    alts = variant_info.get('alts', [])
    label = "positive" if read_base == ref_base or read_base in alts else "negative"
    #print(ref_pos, ref_base, variant_info.get('ref'))
    sequence_around = get_sequence_around(full_read_sequence, read_pos, config)
    print_label(chrom, ref_pos, label, read_base, ref_base, alts, "SNP", sequence_around, config)


"""
Process bases at non-variant loci.
If base not support reference base, then it will be recongnized as sequencing error.
"""
def handle_no_variant_loci(chrom:str, ref_pos:int, read_base:str, ref_base:str,
                           full_read_sequence:str, read_pos:int, config:DictConfig):
    label = "negative" if read_base != ref_base else "positive"
    
    # Here do not output the positive label in non-variant loci
    if label == "positive":
        return
    
    seqeunce_around = get_sequence_around(full_read_sequence, read_pos, config)
    print_label(chrom, ref_pos, label, read_base, ref_base, "None", "None", seqeunce_around, config)


"""
Output labeled data to a file
iterate all bases in read, if the base support reference or variants, it is marked as
positive label. Otherwise it is marked as negative label.
output these information each line:
chrom, position, positive/negative label, read base, reference/variant base, sequence(200 bases before and after the current locus)  
"""
def label_data(chrom: str, read: pysam.AlignedSegment, 
                       variants: dict, reference_sequence:pysam.FastxFile, config:DictConfig):

    # extract the full read sequence
    full_read_sequence = read.query_sequence
    
    # Tract the previous position to identify indels
    indel_sequence, indel_start = "-", 0
    is_indel = False
    last_ref_pos = None

    # Iterate over each base in the read
    for read_pos, ref_pos, _ in read.get_aligned_pairs(with_seq=True):
        if ref_pos is not None:
            last_ref_pos = ref_pos
        
        ref_base = reference_sequence[ref_pos - 1] if ref_pos is not None else None

        if ref_pos is not None and read_pos is not None: # If current base is a matched base
            read_base = full_read_sequence[read_pos - 1]
            
            if is_indel: # If previouse bases exit indel, handle them
                indel_type = "insertion" if indel_sequence.startswith("+") else "deletion"
                # Adjust indel_start for insertions to be the position before the insertion
                corrected_indel_start = indel_start if indel_type == "deletion" else last_ref_pos
                indel_sequence = indel_sequence.replace("+", "").replace("-", "")
                handle_indel_loci(chrom, indel_type, indel_sequence, ref_base,
                             corrected_indel_start, variants, full_read_sequence, config)
                indel_sequence, is_indel = "", False # reset indel tracking
            
            # Handle SNP or matching reference
            if ref_pos in variants:
                handle_snp_loci(chrom, ref_pos, read_base, ref_base, variants,
                                full_read_sequence, read_pos, config)
            else:
                # Handle no-variant loci
                handle_no_variant_loci(chrom, ref_pos, read_base, ref_base,
                                       full_read_sequence, read_pos, config)
                
        
        elif not is_indel: # New start of an indel
            indel_start = last_ref_pos
            is_indel = True
            indel_sequence += ('+' if read_pos is not None else '-') + (full_read_sequence[read_pos] if read_pos is not None else '')
        else: # Continuing an indel
            indel_sequence += ('+' if read_pos is not None else '-') + (full_read_sequence[read_pos] if read_pos is not None else '')
    
    # Check if there was an indel at the end
    if is_indel:
        indel_type = "insertion" if indel_sequence.startswith("+") else "deletion"
        indel_sequence = indel_sequence.replace("+", "").replace("-", "")
        handle_indel_loci(chrom, indel_type, indel_sequence, ref_base,
                     indel_start, variants, full_read_sequence, config)

def generate_label(config):
    bam_file = pysam.AlignmentFile(config.data_path.bam_f)
    reference = load_ref(config.data_path.ref_f)
    confident_regions = load_bed(config.data_path.confident_f) 
    variants_dict = load_vcf(config.data_path.vcf_f)
    #print(variants_dict['chr21'].get(9550563))
    #return 
    
    for read in bam_file:
        if read.mapping_quality <= config.alignment_filter.min_mapping_quality:
            continue # Filter reads with lower mapping quality
        
        chrom = bam_file.get_reference_name(read.reference_id)
        if chrom in confident_regions and is_read_in_confident_region(read, confident_regions[chrom]):
            #new_label_data(chrom, read, variants_dict[chrom], reference[chrom], config)
            label_data(chrom, read, variants_dict[chrom], reference[chrom], config)
    

@hydra.main(version_base=None, config_path='../../configs/error-prediction', config_name='params.yaml')
def main(config: DictConfig) -> None:
    generate_label(config) 


if __name__ == '__main__':
    main()