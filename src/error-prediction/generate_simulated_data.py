

def print_label(chrom:str, type:str, read_strand:str, position:int, label:str, 
                read_base:str, ref_base:str, alts:str,
                is_germline:str, variant_type:str,
                sequence_around:str, file_p:str):
    """
    Print label to output config.data_path_label_f file.
    position should be change to 1-base
    """
    label_f = open(file_p, "a")
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

def main():
    file_p = '/home/yang1031/projects/error-aware-calling/data/illumina/simulate_data_insertion'
    tokens = ['A','C','G','T','-']
    for token in tokens:
        index = tokens.index(token)
        read_base = token * (index + 2)
        for i in range(20000):
            print_label(chrom='chr',
                        type='Insertion',
                        read_strand='forward',
                        position=1111111,
                        label=token,
                        read_base=read_base,
                        ref_base=token,
                        alts='[]',
                        is_germline="False",
                        variant_type="Insertion",
                        sequence_around=token*100,
                        file_p=file_p)
    
    
if __name__ == "__main__":
    main()