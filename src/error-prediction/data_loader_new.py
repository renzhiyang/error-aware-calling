import pysam
import hydra
import gzip

from omegaconf import DictConfig, OmegaConf
from typing import List, Dict

# Load phased variants
class PhasedVCFReader:
    def __init__(self, vcf_file):
        self.vcf_file = vcf_file
        self.variants = self.load_variants()
    
    def load_variants(self):
        variants = {}
        open_func = gzip.open if self.vcf_file.endswith('.gz') else open
        total_count = 0
        phased_count = 0
        with open_func(self.vcf_file, 'rt') as file:
            for line in file:
                if line.startswith('#'):
                    continue # skip header lines
                columns = line.strip().split("\t")
                genotype = columns[-1].split(':')[0]
                phased = True if genotype[1] == "|" else False
                if not phased: 
                    continue # don't consider variant that is not phased
                
                chr = columns[0]
                pos = int(columns[1])
                ref = columns[3]
                alts = columns[4].split(',')
                is_indel = len(ref) > 1 or any(len(alt)>1 for alt in alts) 
                ref_alts = [ref] + alts
                
                haplotype = [ref_alts[int(genotype[0])], ref_alts[int(genotype[-1])]]
                if chr not in variants:
                    variants[chr] = {}
                variants[chr][pos] = {'ref_alts': ref_alts, 'haplotype': haplotype, 'is_indel': is_indel}
        return variants
    
    def get_variants(self):
        return self.variants
            
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

# To determine if the current read is phased or not
def get_phased_read_haplotype(read: pysam.AlignedSegment):
    haplotype = None
    for tag, value in read.get_tags():
        if tag == 'HP':
            haplotype = value
            break
    return haplotype


def get_sequence_around(full_read_sequence:str, position:int, 
                        insert_len:int , is_del:bool, config:DictConfig):
    """
    Extract the sequence around a given position with a specified window size.
    example: half window is 2
    SNV: return AACTT, C is the variant posision, 1 window + 1 base
    Insertion: true AATT, query AAAATT insert 2 bases, return AAAATT, 1 window + insert bases
    Deletion: true AACCTT, query AATT delete 2 bases, return AATT, 1 window size
    """
    window_half = config.label.window_size_half
    start = max(0, position - window_half)
    end = min(len(full_read_sequence), position + insert_len + window_half + 1)
    if is_del:
        return full_read_sequence[start:end-1]
    else:
        return full_read_sequence[start:end]


def print_label(chrom:str, position:int, label:str, 
                read_base:str, ref_base:str, alts:str,
                variant_type:str, non_variant_type:str,
                sequence_around:str, config:DictConfig):
    """
    Print label to output config.data_path_label_f file.
    position should be change to 1-base
    """
    
    label_f = open(config.data_path.label_f, "a")
    print(
    f'chrom:{chrom} '
    f'pos_ref:{position + 1} '
    f'label:{label} '
    f'read_base:{read_base} '
    f'ref_base:{ref_base} '
    f'alts:{alts} '
    f'variant_type:{variant_type} '
    f'non_variant_type:{non_variant_type} '
    f'sequence_around:{sequence_around} ',
    flush=True, file=label_f
)


def label_unphased_read(chrom:str, read:pysam.AlignedSegment, 
                        ref_seq:pysam.FastxFile,
                        phased_variants:dict, config:DictConfig):
    """
    For unphased read
    1. exclude germline variant locus
    2. only label negative samples in non-germline variant locus
    
    Find differences between the read and the reference sequence from an aligned segment.

    Args:
    aligned_segment (pysam.AlignedSegment): An aligned segment from a BAM or SAM file.

    Returns:
    
    """
    full_read_sequence = read.query_sequence
    ref_pos = read.reference_start  
    read_pos = 0
    
    for cigar_tuple in read.cigar:
        cigar_op = cigar_tuple[0]
        length = cigar_tuple[1]
        
        if cigar_op == 0: # match pr mismatch
            for i in range(length):
                if ref_pos + i >= len(ref_seq) or read_pos + i >= len(full_read_sequence):
                    continue  # Avoid index out of range
                ref_base = ref_seq[ref_pos + i]
                read_base = full_read_sequence[read_pos + i]
                # mismatch, and non germline variant loci, vcf file variant should change to 1-based.
                if ref_base != read_base and ref_pos + 1 not in phased_variants: 
                    sequence_around = get_sequence_around(full_read_sequence, read_pos + i, 0, False, config)
                    print_label(chrom, ref_pos + i, ref_base, read_base, ref_base, 'None', 
                                'None', "SNV", sequence_around, config)
            ref_pos += length
            read_pos += length
        
        elif cigar_op == 1: # insertion to the reference
            inserted_bases = full_read_sequence[read_pos:read_pos + length]
            sequence_around = get_sequence_around(full_read_sequence=full_read_sequence, 
                                                  position=read_pos, 
                                                  insert_len=len(inserted_bases), 
                                                  is_del=False,
                                                  config=config)
            print_label(chrom, ref_pos, "-", inserted_bases, "-", "None", 
                        "None", "insertion", sequence_around, config)
            #print("insertion", ref_pos, inserted_bases)
            read_pos += length
        
        elif cigar_op == 2: # deletion from the reference
            deleted_bases = ref_seq[ref_pos:ref_pos + length]
            sequence_around = get_sequence_around(full_read_sequence=full_read_sequence, 
                                                  position=read_pos, 
                                                  insert_len=0, 
                                                  is_del=True, 
                                                  config=config)
            print_label(chrom, ref_pos, deleted_bases, "-", deleted_bases,
                        "None", "None", "deletion", sequence_around, config)
            print("deletion", ref_pos, deleted_bases)
            ref_pos += length
            
        elif cigar_op == 3: # N (skipped region from the reference) 
            ref_pos += length
        
        elif cigar_op == 4: # S (soft clipping)
            read_pos += length

        # No need to adjust  fro H (hard clipping) and P (padding) as they don't consume ref or read
     
        
def label_phased_read():
    return


def label_data(chrom: str, read: pysam.AlignedSegment, 
                       phased_variants: dict, ref_seq:pysam.FastxFile, config:DictConfig):
    haplotype = get_phased_read_haplotype(read)
    if haplotype == None:
        label_unphased_read(chrom, read, ref_seq, phased_variants, config)
        return
    label_phased_read()


def generate_label(config):
    #bam_file = pysam.AlignmentFile(config.data_path.bam_f)
    vcf_reader = PhasedVCFReader(config.data_path.phased_vcf)
    phased_variants = vcf_reader.get_variants()
    bam_file = pysam.AlignmentFile(config.data_path.tagged_bam)
    reference = load_ref(config.data_path.ref_f)
    confident_regions = load_bed(config.data_path.confident_f) 

    for read in bam_file:
        if read.mapping_quality <= config.alignment_filter.min_mapping_quality:
            continue # Filter reads with lower mapping quality
        
        chrom = bam_file.get_reference_name(read.reference_id)
        if chrom in confident_regions and is_read_in_confident_region(read, confident_regions[chrom]):
            #new_label_data(chrom, read, variants_dict[chrom], reference[chrom], config)
            label_data(chrom, read, phased_variants[chrom], reference[chrom], config)
    
    
@hydra.main(version_base=None, config_path='../../configs/error-prediction', config_name='params.yaml')
def main(config: DictConfig) -> None:
    generate_label(config) 


if __name__ == '__main__':
    main()