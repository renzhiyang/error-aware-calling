#!/bin/bash

usage() {
    echo "Usage: ./genotype_to_vcf.sh -r <ref_file> --geno_folder <genotype folder> --vcf_folder <vcf folder> -t <slots>"
    echo "Example: ./run.sh -b input.bam -r reference.bam -o output_pileup -t 4"
    echo 'Required parameters:'
    echo '  -r, --ref_fn=FILE            Reference file input.'
    echo '  -o, --vcf_folder=FILE     Output file prefix.'
    echo '  -t, --slots=INT              Number of slots to use.'
    echo '  -g, --geno_folder=FILE       Genotype folder.'
    echo ''
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
    -r | --ref_fn)
        REF_FILE="$2"
        shift
        ;;
    -o | --vcf_folder)
        OUTPUT_PREFIX="$2"
        shift
        ;;
    -g | --geno_folder)
        GENO_FOLDER="$2"
        shift
        ;;
    -t | --slots)
        SLOTS="$2"
        shift
        ;;
    esac
    shift
done

convert_geno_to_vcf(){
    local geno_file="$1"
    vcf_filename="${geno_file##*/}"
    python -m src.genotype_to_vcf \
        --ref_fn "$REF_FILE" \
        --geno_fn "$geno_file" \
        --vcf_fn "$OUTPUT_PREFIX/$vcf_filename" \
        --ctg_name "chr20"

    #bgzip "$OUTPUT_PREFIX/$vcf_filename"
    #tabix -p vcf "$OUTPUT_PREFIX/$vcf_filename.gz"
}

export -f convert_geno_to_vcf
export REF_FILE OUTPUT_PREFIX GENO_FOLDER SLOTS

find "$GENO_FOLDER" -name "*.vcf" | parallel --jobs "$SLOTS" convert_geno_to_vcf