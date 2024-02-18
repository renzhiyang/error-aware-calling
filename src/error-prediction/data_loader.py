import pysam
import hydra
import gzip

from omegaconf import DictConfig, OmegaConf

def read_vcf(filename):
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
            variants[pos] = {'chr': chr, 'ref': ref, 'alts': alts, 'is_indel': is_indel}
    return variants

# The function to identify if the current base in sequencing read support ref/variant or not.
# If the current base not support ref/variant, it will be recongnized as sequencing error.
# It only work in highly condident regions, errors are used to build training data
def is_error(position, base, is_indel):
    return None

def build_training_label(cfg):
    bam_file = pysam.AlignmentFile(cfg.data_path.bam_f)
    for read in bam_file:
        if read.mapping_quality <= cfg.alignment.min_mapping_quality:
            continue
        return
    
    

@hydra.main(version_base=None, config_path='../../configs/error-prediction', config_name='params.yaml')
def main(cfg: DictConfig) -> None:
    build_training_label(cfg)
    return 
    variants = read_vcf(cfg.data_path.vcf_f)


if __name__ == '__main__':
    main()
    
            