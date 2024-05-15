import os
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
                #if not phased: 
                #    continue # don't consider variant that is not phased
                
                chr = columns[0]
                pos = int(columns[1])
                ref = columns[3]
                alts = columns[4].split(',')
                is_indel = len(ref) > 1 or any(len(alt)>1 for alt in alts) 
                ref_alts = [ref] + alts
                
                haplotype = [ref_alts[int(genotype[0])], ref_alts[int(genotype[-1])]]
                if chr not in variants:
                    variants[chr] = {}
                if phased:
                    variants[chr][pos] = {'ref_alts': ref_alts, 'haplotype': haplotype, 'is_indel': is_indel, 'phased': True}
                else:
                    variants[chr][pos] = {'ref_alts': ref_alts, 'haplotype': haplotype, 'is_indel': is_indel, 'phased': False}
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

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_ref(filename: str):
    reference = {}
    with pysam.FastxFile(filename) as fh:
        for entry in fh:
            reference[entry.name] = entry.sequence
    return reference

def reverse_seq(sequence: str):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    if str is not None:
        return ''.join(complement[base] for base in sequence[::-1])
    else:
        return sequence

# To verify if read in confident regions
def is_read_in_confident_region(read: pysam.AlignedSegment, confident_regions: list) -> bool:
    for start, end in confident_regions:
        if read.reference_start < end and read.reference_end > start:
            return True
    return False

def is_pos_in_confident_region(pos: int, confident_regions: list) -> bool:
    for start, end in confident_regions:
        if pos > start and pos < end:
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

def get_sequence(full_read_sequence:str, position:int, is_forward:bool,
                        insert_len:int , is_indel:bool, config:DictConfig):
    """
    Extract the sequence around a given position with a specified window size.
    example: half window is 2
    SNV: return AACTT, C is the variant posision, 1 window + 1 base
    Insertion: true AATT, query AAAATT insert 2 bases
        around return AAAATT, forward return AA 1 window + insert bases
        read_base AA
    Deletion: true AACCTT, query AATT delete 2 bases, 
        around return AATT, forward return AA, 1 window size
    """
    window_half = config.label.window_size_half
    
    # if the current read is reverse strand, update the current position
    if not is_forward:
        position = len(full_read_sequence) - position - insert_len - 1
        if is_indel:
            position += 1

    start = max(0, position - window_half)
    # 是否是输出前面的bases，或者是前后的bases
    end = min(len(full_read_sequence), position + insert_len + window_half + 1)
    if not config.controller.seq_around == True:
        end = min(len(full_read_sequence), position + 1)
    
    # previous version
    if is_indel:
        return full_read_sequence[start:end-1]
    else:
        return full_read_sequence[start:end]
    
    # new version
    #if is_forward:
    #    if is_indel:
            

def get_tag(read, tag_name):
    for tag in read.tags:
        if tag[0] == tag_name:
            print(tag[1])
            return tag[1]
    return None

def print_label(chrom:str, type:str, read_strand:str, position:int, label:str, 
                read_base:str, ref_base:str, alts:str,
                is_germline:str, variant_type:str,
                sequence_around:str, config:DictConfig):
    """
    Print label to output config.data_path_label_f file.
    position should be change to 1-base
    """

    # don't output the first forward bases (illumina: 100, ONT: 256)
    if len(sequence_around) < config.label.window_size_half:
        return

    # reverse read base
    #if read_strand == "reverse":
    #    print(f'position:{position}, read base:{read_base}')
    #    read_base = reverse_seq(read_base)
    
    label_f = open(config.data_path.label_f, "a")
    print(
    f'chrom:{chrom} '
    f'type:{type} '
    f'read_strand:{read_strand} '
    f'pos_ref:{position} '
    f'label:{label} '
    f'read_base:{read_base} '
    f'ref_base:{ref_base} '
    f'alts:{alts} '
    f'is_germline:{is_germline} '
    f'variant_type:{variant_type} '
    f'sequence_around:{sequence_around} ',
    flush=True, file=label_f
)

def label_unphased_read(chrom:str, read:pysam.AlignedSegment, 
                        ref_seq:str,
                        variants:dict, 
                        confident_regions: list, 
                        config:DictConfig):
    """
    For unphased read
    1. exclude germline variant locus
    2. only label negative samples in non-germline variant locus
    
    Find differences between the read and the reference sequence from an aligned segment.

    Args:
    aligned_segment (pysam.AlignedSegment): An aligned segment from a BAM or SAM file.

    Returns:
    
    """
    # Output the current query name and query length
    print(f'Unphased read: {read.query_name}, length: {read.query_length}', flush=True)
    
    # Track the progress of program
    current_pos = 0
    label_count = 0
    tracking_threshold = {30, 60, 90}
    
    full_read_sequence = read.query_sequence # 得到的是align后的序列，也就是在IGV可以看到的序列，有可能是反向的
    forward_sequence = read.get_forward_sequence() # 得到的是完全正向的测序序列
    read_strand = "forward" if read.is_forward else "reverse"
    ref_pos = read.reference_start  
    read_pos = 0
    
    for cigar_tuple in read.cigar:
        cigar_op = cigar_tuple[0]
        length = cigar_tuple[1]
        if cigar_op == 0: # match pr mismatch
            for i in range(length):
                if ref_pos + i >= len(ref_seq) or read_pos + i >= len(full_read_sequence): 
                    continue  # Avoid index out of range
                if not is_pos_in_confident_region(ref_pos + i, confident_regions):
                    continue
                
                ref_base = ref_seq[ref_pos + i]
                read_base = full_read_sequence[read_pos + i]
                # mismatch, and non germline variant loci, vcf file variant should change to 1-based.
                if ref_base != read_base and ref_pos + i + 1  not in variants: 
                    sequence_around = get_sequence(full_read_sequence=forward_sequence, 
                                                          position=read_pos + i, 
                                                          is_forward = read.is_forward,
                                                          insert_len=0, 
                                                          is_indel=False, 
                                                          config=config)
                    print_label(chrom=chrom, type='Negative', read_strand=read_strand,
                                position=ref_pos+i+1, label=ref_base, 
                                read_base=read_base, ref_base=ref_base, 
                                alts=None, is_germline="No", variant_type="SNV", 
                                sequence_around=sequence_around, config=config)
                    label_count += 1
                    # Print test
                    #print(f'ref_pos:{ref_pos+i+1}, ref_base:{ref_base}, read_base:{read_base}'
                    #  f'sequence_around:{sequence_around}')
            ref_pos += length
            read_pos += length
        
        elif cigar_op == 1: # insertion to the reference
            ref_base = ref_seq[ref_pos - 1]
            read_base = full_read_sequence[read_pos - 1]
            inserted_bases = full_read_sequence[read_pos:read_pos + length]
            is_germline = True if ref_pos in variants else False
            
            if not is_pos_in_confident_region(ref_pos, confident_regions):
                read_pos += length
                continue
            
            if not is_germline:
                combined = read_base + inserted_bases
                sequence_around = get_sequence(full_read_sequence=forward_sequence, 
                                                    position=read_pos, 
                                                    is_forward = read.is_forward,
                                                    insert_len=len(inserted_bases), 
                                                    is_indel=True,
                                                    config=config)
                print_label(chrom=chrom, type='Negative', read_strand=read_strand,
                            position=ref_pos, label=ref_base, 
                            read_base=combined, ref_base=ref_base, 
                            alts=None, is_germline="No", variant_type="Insertion", 
                            sequence_around=sequence_around, config=config)
                label_count += 1
                # Print test
                #print(f'ref_pos:{ref_pos}, ref_base:{ref_base}, inserted_bases:{inserted_bases}, sequence_around:{sequence_around}')
            read_pos += length     
        
        elif cigar_op == 2: # deletion from the reference
            deleted_bases = ref_seq[ref_pos:ref_pos + length]
            ref_base = ref_seq[ref_pos - 1]
            read_base = full_read_sequence[read_pos - 1]
            is_germline = True if ref_pos in variants else False
            
            if not is_pos_in_confident_region(ref_pos, confident_regions):
                ref_pos += length
                continue
            
            if not is_germline:
                sequence_around = get_sequence(full_read_sequence=forward_sequence, 
                                                  position=read_pos, 
                                                  is_forward = read.is_forward,
                                                  insert_len=0, 
                                                  is_indel=True,
                                                  config=config)
                print_label(chrom=chrom, type='Negative', read_strand=read_strand,
                            position=ref_pos, label=ref_base+deleted_bases, 
                            read_base=read_base + len(deleted_bases)*'-', 
                            ref_base=ref_base+deleted_bases, 
                            alts=None, is_germline="No", variant_type="Deletion", 
                            sequence_around=sequence_around, config=config)
                label_count += 1
                # Print test
                #print(f'ref_pos:{ref_pos}, ref_base:{ref_base+deleted_bases}, deleted_bases:{deleted_bases}, sequence_around:{sequence_around}')
            ref_pos += length
            
        elif cigar_op == 3: # N (skipped region from the reference) 
            ref_pos += length
        
        elif cigar_op == 4: # S (soft clipping)
            read_pos += length

        # No need to adjust  fro H (hard clipping) and P (padding) as they don't consume ref or read
        
        # Update tracking
        current_pos += length
        progress = (current_pos / read.query_length) * 100
        crossed_threshold = {t for t in tracking_threshold if progress >= t}
        for t in crossed_threshold:
            #print(f"Crossed {t}% threshold at position: {current_pos}, ref pos: {ref_pos}, Progress: {progress:.2f}%, labels: {label_count}", flush=True)
            tracking_threshold.remove(t)
         
        
def label_phased_read(chrom:str, read:pysam.AlignedSegment, 
                        ref_seq:pysam.FastxFile,
                        variants:dict, 
                        confident_regions: list, 
                        config:DictConfig):
    # Output the current query name and query length
    #print(f'Phased read: {read.query_name}, length: {read.query_length}', flush=True)
    # Track the progress of program
    current_pos = 0
    label_count = 0
    tracking_threshold = {30, 60, 90}
    
    full_read_sequence = read.query_sequence # 得到的是align后的序列，也就是在IGV可以看到的序列，有可能是反向的
    forward_sequence = read.get_forward_sequence() # 得到的是完全正向的测序序列
    read_strand = "forward" if read.is_forward else "reverse"
    ref_pos = read.reference_start
    haplotype_index = get_tag(read, "HP") - 1
    read_pos = 0
    
    for cigar_tuple in read.cigar:
        cigar_op = cigar_tuple[0]
        length = cigar_tuple[1]
        
        if cigar_op == 0: # match or mismatch
            # if read_base != label, then choosed as negative label
            # if read_base == label in phased variants, then choosed as positive label
            for i in range(length - 1): # match的以后一个是indel的开端，因此这个位点放到indel里去处理
                if not is_pos_in_confident_region(ref_pos + i, confident_regions):
                    # exclude possitions not in confident regions
                    continue
                if ref_pos + i >= len(ref_seq) or read_pos + i >= len(full_read_sequence):
                    # avoid index out of range
                    continue 
                
                ref_base = ref_seq[ref_pos + i]
                read_base = full_read_sequence[read_pos + i]
                is_germline = True if ref_pos + i + 1 in variants else False
                label = ref_base
                alts = None
                
                if not is_germline and read_base != label:
                    # 对于不是germline variant的位点，只取negative，即read base和reference base不同
                    # read base和reference base相同的情况直接跳过 
                    # 需要用到forward_sequence, 完全正向的测序序列
                    sequence_around = get_sequence(full_read_sequence=forward_sequence, 
                                                          position=read_pos + i,
                                                          is_forward = read.is_forward,
                                                          insert_len=0,
                                                          is_indel=False,
                                                          config=config)
                    # 注意：因为打印出来的是要1-base的position，因此要 ref_pos+i+1
                    # 但是Indel的情况是不需要的，因为在记录Indel的时候，往往是以插入或删除块的前一个位点为基准
                    print_label(chrom=chrom, type='Negative', read_strand=read_strand,
                                position=ref_pos+i+1, label=label, 
                                read_base=read_base, ref_base=ref_base, 
                                alts=alts, is_germline="No", variant_type="SNV", 
                                sequence_around=sequence_around, config=config)
                    label_count += 1
                
                if is_germline:
                    #对于是germline variant的位点，还要分两种情况：1.phased variant；2.unphased variant
                    #1. phased variant: positive和negative分别是read base和haplotype base相同和不同
                    #2. unphased variant:排除在外，不考虑这种情况，直接跳过
                    
                    # 补充情况: 对于类似ch1:40702341的情况，haplotypes: CT, C. label: CT, 此时的read base
                    # 是C，按照之前的算法会被判定为是Negative SNV；因此还需要看后几个base是否和label相同
                    # 如果后面是T，那么CT=label，是正确的
                    # 对于这种请款，CT应该放入deletion的情况中处理，这里只处理SNV，因此需要判断read和label长度是否一致
                    is_phased_variant = variants[ref_pos + i + 1]['phased']
                    label = variants[ref_pos + i + 1]['haplotype'][haplotype_index] # variants are 1-based
                    alts = variants[ref_pos + i + 1]['ref_alts']
                           
                    sequence_around = get_sequence(full_read_sequence=forward_sequence,  # type: ignore
                                                          position=read_pos + i,
                                                          is_forward = read.is_forward,
                                                          insert_len=0,
                                                          is_indel=False,
                                                          config=config)
                    if is_phased_variant:
                        if read_base == label:
                            print_label(chrom=chrom, type='Positive', read_strand=read_strand, 
                                        position=ref_pos+i+1, label=label, 
                                        read_base=read_base, ref_base=ref_base, 
                                        alts=alts, is_germline="Yes", variant_type="SNV", 
                                        sequence_around=sequence_around, config=config)
                        elif len(read_base) == len(label):
                            #必须要长度相同的情况下，才可以作为SNV
                            print_label(chrom=chrom, type='Negative', read_strand=read_strand,
                                        position=ref_pos+i+1, label=label, 
                                        read_base=read_base, ref_base=ref_base, 
                                        alts=alts, is_germline="Yes", variant_type="SNV", 
                                        sequence_around=sequence_around, config=config)
                        label_count += 1
            ref_pos += length
            read_pos += length
        
        elif cigar_op == 1: # insertion to the reference    
            inserted_bases = full_read_sequence[read_pos: read_pos + length]
            read_base = full_read_sequence[read_pos - 1]
            ref_base = ref_seq[ref_pos - 1] # insertion是以插入bases的前一个位点为基准
            is_germline = True if ref_pos in variants else False
            alts = None
            
            if not is_pos_in_confident_region(ref_pos, confident_regions):
                # 不在confident region中
                read_pos += length
                continue
            
            if is_germline:
                ''' 
                    如果是germline variant位点的话,分为以下几种情况:
                    1. phased variant: 得到 positive 和 nagative的训练集
                    2. unphased variant: 直接跳过
                '''
                is_phased_variant = variants[ref_pos]['phased']
                label = variants[ref_pos]['haplotype'][haplotype_index]
                alts = variants[ref_pos]['ref_alts']
                sequence_around = get_sequence(full_read_sequence=forward_sequence, 
                                                  position=read_pos, 
                                                  is_forward = read.is_forward,
                                                  insert_len=len(inserted_bases), 
                                                  is_indel=True,
                                                  config=config)
                if is_phased_variant:
                    combined = read_base + inserted_bases
                    if combined == label:
                        print_label(chrom=chrom, type='Positive', read_strand=read_strand,
                                    position=ref_pos, label=label, 
                                    read_base=combined, ref_base=ref_base, 
                                    alts=alts, is_germline="Yes", variant_type="Insertion", 
                                    sequence_around=sequence_around, config=config)
                    elif len(combined) != len(label): 
                        #排除掉insertion中有测序错误的情况，只是标记insertion error
                        print_label(chrom=chrom, type='Negative', read_strand=read_strand,
                                    position=ref_pos, label=label, 
                                    read_base=combined, ref_base=ref_base, 
                                    alts=alts, is_germline="Yes", variant_type="Insertion", 
                                    sequence_around=sequence_around, config=config)
                    label_count += 1
                        
                # Test print
                #print(f'ref_pos:{ref_pos}, ref_base:{ref_base}, label:{label}, alts:{alts}'
                #      f'inserted_base:{inserted_bases}, is_phased_variant:{is_phased_variant}'
                #      f'sequence_around:{sequence_around}')
                
            else:
                '''如果不是germline variant位点的话,被视为是insertion sequencing error'''
                combined = read_base + inserted_bases
                sequence_around = get_sequence(full_read_sequence=forward_sequence, 
                                                  position=read_pos, 
                                                  is_forward = read.is_forward,
                                                  insert_len=len(inserted_bases), 
                                                  is_indel=True,
                                                  config=config)
                print_label(chrom=chrom, type='Negative', read_strand=read_strand,
                            position=ref_pos, label=ref_base, 
                            read_base=combined, ref_base=ref_base, 
                            alts=alts, is_germline="No", variant_type="Insertion", 
                            sequence_around=sequence_around, config=config)
                label_count += 1
                # Test print
                #print(f'ref_pos:{ref_pos}, ref_base:{ref_base}, inserted_base:{inserted_bases}',
                #      f'sequence_around:{sequence_around}')   
            read_pos += length
        
        elif cigar_op == 2: # deletion from the reference
            deleted_bases = ref_seq[ref_pos:ref_pos + length]
            ref_base = ref_seq[ref_pos - 1] # deletion同样也是以前一个位置为基点
            read_base = full_read_sequence[read_pos - 1]
            
            is_germline = True if ref_pos  in variants else False
            alts = None
            
            if not is_pos_in_confident_region(ref_pos, confident_regions):
                ref_pos += length
                continue
            
            if is_germline:
                '''
                   如果是germline variants的话, 处理方式和之前相同，分为两种情况:
                   1. phased variant: 得到 positive 和 negative 的训练集
                   例如 phased variant: hap_1 ATTTT, hap_2 A. 
                   当前read是 ATTT, 并且是属于hap_1, 那么是negative
                   2. unphased variant: 直接跳过
                '''
                is_phased_variant = variants[ref_pos]['phased']
                label = variants[ref_pos]['haplotype'][haplotype_index]
                another_hap = variants[ref_pos]['haplotype'][1 - haplotype_index]
                alts = variants[ref_pos]['ref_alts']
                sequence_around = get_sequence(full_read_sequence=forward_sequence, 
                                                  position=read_pos, 
                                                  is_forward = read.is_forward,
                                                  insert_len=0, 
                                                  is_indel=True,
                                                  config=config)
                
                if is_phased_variant:
                    '''
                    #TODO: 不够完善, 目前还是只提取出positive的samples
                    Positive的情况: len(deleted_bases) == len(another_hap) - len(label)
                    Negative的情况: 
                        1. read_base != label and len(read_base) != len(label) 
                    example: 
                        haps ['AACTT', 'A'], 
                        hap1: AACTT, read: A + any number deletion ---- Negative
                        hap2: A, read: A + 4del ---- Positive
                        hap2: A, read: A + other number del ---- Negative
                    '''
                    if len(deleted_bases) == len(another_hap) - len(label):
                        print_label(chrom=chrom, type='Positive', read_strand=read_strand,
                                    position=ref_pos, label=label, 
                                    read_base=read_base + len(deleted_bases)*'-', 
                                    ref_base=ref_base+deleted_bases, 
                                    alts=alts, is_germline="Yes", variant_type="Deletion", 
                                    sequence_around=sequence_around, config=config)
                    #elif:
                    #    print_label(chrom=chrom, type='Negative', read_strand=read_strand,
                    #                position=ref_pos, label=label, 
                    #                read_base=read_base + len(deleted_bases)*'-', 
                    #                ref_base=ref_base+deleted_bases, 
                    #                alts=alts, is_germline="Yes", variant_type="Deletion", 
                    #                sequence_around=sequence_around, config=config)
                    '''
                    # previous version, with bugs
                    if ref_base + deleted_bases == label:
                        print_label(chrom=chrom, type='Positive', read_strand=read_strand,
                                    position=ref_pos, label=label, 
                                    read_base=read_base, ref_base=ref_base+deleted_bases, 
                                    alts=alts, is_germline="Yes", variant_type="Deletion", 
                                    sequence_around=sequence_around, config=config)
                    else:
                        print_label(chrom=chrom, type='Negative', read_strand=read_strand,
                                    position=ref_pos, label=label, 
                                    read_base=read_base, ref_base=ref_base+deleted_bases, 
                                    alts=alts, is_germline="Yes", variant_type="Deletion", 
                                    sequence_around=sequence_around, config=config)
                    '''
                    label_count += 1
                    # print test
                    #print(f'ref_pos:{ref_pos}, ref_base:{ref_base}, label:{label}, alts:{alts}, deleted_bases:{deleted_bases}'
                    #    f'sequence:{sequence_around}')
            else:
                '''
                    如果不是germline variant位点, 则视为sequencing error
                '''
                sequence_around = get_sequence(full_read_sequence=forward_sequence, 
                                                  position=read_pos, 
                                                  is_forward = read.is_forward,
                                                  insert_len=0, 
                                                  is_indel=True,
                                                  config=config)
                print_label(chrom=chrom, type='Negative', read_strand=read_strand,
                            position=ref_pos, label=ref_base+deleted_bases, 
                            read_base=read_base + len(deleted_bases)*'-', 
                            ref_base=ref_base+deleted_bases, 
                            alts=alts, is_germline="No", variant_type="Deletion", 
                            sequence_around=sequence_around, config=config)
                label_count += 1
                # print test
                #print(f'ref_pos:{ref_pos}, ref_base:{ref_base}, read_base:{read_base}, label:{ref_base + deleted_bases}, alts:{alts}, deleted_bases:{deleted_bases} '
                #        f'sequence:{sequence_around}')
                
                
            ref_pos += length
        
        elif cigar_op == 3:
            ref_pos += length
        
        elif cigar_op == 4:
            read_pos += length
        
        # Update tracking
        current_pos += length
        progress = (current_pos / read.query_length) * 100
        crossed_threshold = {t for t in tracking_threshold if progress >= t}
        for t in crossed_threshold:
            #print(f"Crossed {t}% threshold at position: {current_pos}, Progress: {progress:.2f}%, labels: {label_count}", flush=True)
            tracking_threshold.remove(t)
    
    return 


def label_data(chrom: str, read: pysam.AlignedSegment, 
                       variants: dict, ref_seq:pysam.FastxFile, 
                       confident_regions: list, config:DictConfig):
    haplotype = get_phased_read_haplotype(read)
    #print(type(config.controller.unphased),config.controller.unphased)
    if haplotype == None and config.controller.unphased == True:
        label_unphased_read(chrom, read, ref_seq, variants, confident_regions, config)
    elif haplotype != None and config.controller.phased == True:
        label_phased_read(chrom, read, ref_seq, variants, confident_regions, config)
    else:
        return


def generate_label(config):
    #bam_file = pysam.AlignmentFile(config.data_path.bam_f)
    vcf_reader = PhasedVCFReader(config.data_path.phased_vcf)
    variants = vcf_reader.get_variants()
    bam_file = pysam.AlignmentFile(config.data_path.tagged_bam)
    reference = load_ref(config.data_path.ref_f)
    confident_regions = load_bed(config.data_path.confident_f) 

    for read in bam_file:
        if read.mapping_quality <= config.alignment_filter.min_mapping_quality:
            continue # Filter reads with lower mapping quality
        
        chrom = bam_file.get_reference_name(read.reference_id)
        
        if chrom in confident_regions and is_read_in_confident_region(read, confident_regions[chrom]) :
            #new_label_data(chrom, read, variants_dict[chrom], reference[chrom], config)
            label_data(chrom, read, variants[chrom], reference[chrom], confident_regions[chrom], config)
    
@hydra.main(version_base=None, config_path='../../configs', config_name='defaults.yaml')
def main(config: DictConfig) -> None:
    config = config.label_data
    #print(config.label.window_size_half, config.data_path.label_f, flush=True)
    print(OmegaConf.to_yaml(config), flush=True)
    generate_label(config) 

if __name__ == '__main__':
    main()