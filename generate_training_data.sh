#!/bin/bash

usage() {
  echo "Usage: ./run.sh -b <bam_file> -r <ref_file> -o <output_prefix> -t <slots>"
  echo "Example: ./run.sh -b input.bam -r reference.bam -o output_pileup -t 4"
  echo 'Required parameters:'
  echo '  -b, --bam_fn=FILE            BAM file input. The input file must be samtools indexed.'
  echo '  -r, --ref_fn=FILE            Reference file input.'
  echo '  -o, --output_prefix=FILE     Output file prefix.'
  echo '  -t, --slots=INT              Number of slots to use.'
  echo '  --bed_fn=FILE                The regions that will be processed.'
  echo ' --vcf_fn=FILE                The phased VCF file.'
  echo ''
  echo 'Optional parameters:'
  echo '  --chunk_size=INT            The chunk size for parallel processing.'
  echo '  --samtools=STR               Path to samtools.'
  echo '  -win_s, --window_size_half                   the input window size'
  exit 1
}

version() {
  echo "generate_training_data.sh version 1.0"
}

# Check if no arguments are passed
if [ $# -eq 0 ]; then
  usage
fi

# Default values
# CHUNK_SIZE=1000000
SAMTOOLS="samtools"
WINDOW_SIZE_HALF=41

########### Parse command line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
  -b | --bam_fn)
    BAM_FILE="$2"
    shift
    ;;
  -r | --ref_fn)
    REF_FILE="$2"
    shift
    ;;
  -o | --output_prefix)
    OUTPUT_PREFIX="$2"
    shift
    ;;
  -t | --slots)
    SLOTS="$2"
    shift
    ;;
  --bed_fn)
    BED_FILE="$2"
    shift
    ;;
  --chunk_size)
    CHUNK_SIZE="$2"
    shift
    ;;
  --bed_fn)
    BED_FILE="$2"
    shift
    ;;
  --vcf_fn)
    VCF_FILE="$2"
    shift
    ;;
  --window_size_half)
    WINDOW_SIZE_HALF="$2"
    shift
    ;;
  --samtools)
    SAMTOOLS="$2"
    shift
    ;;

  --)
    shift
    break
    ;;
  -h | --help)
    usage
    exit 0
    ;;
  -v | --version)
    version
    exit 0
    ;;
  *)
    echo "Unknown parameter passed: $1"
    usage
    ;;
  esac
  shift
done

# Check if required parameters are provided
if [ -z "$VCF_FILE" ] || [ -z "$BED_FILE"] || [ -z "$BAM_FILE" ] || [ -z "$REF_FILE" ] || [ -z "$OUTPUT_PREFIX" ] || [ -z "$SLOTS" ]; then
  echo "Error: Missing required parameters."
  usage
fi

############### Confirm the inputs
echo "Input Parameters:"
echo "BAM File: $BAM_FILE"
echo "Reference File: $REF_FILE"
echo "Output Prefix: $OUTPUT_PREFIX"
echo "Slots: $SLOTS"
echo "BED File: $BED_FILE"
echo "VCF File: $VCF_FILE"
echo "Window Size Half: $WINDOW_SIZE_HALF"
echo "Chunk Size: $CHUNK_SIZE"
echo ""

############### Check if required files exits
if [ ! -f "$BAM_FILE" ]; then
  echo "Error: BAM file '$BAM_FILE' does not exist."
  exit 1
fi

if [ ! -f "$REF_FILE" ]; then
  echo "Error: Reference file '$REF_FILE' does not exist."
  exit 1
fi

if [ ! -f "$BED_FILE" ]; then
  echo "Error: BED file '$BED_FILE' does not exist."
  exit 1
fi

if [ ! -f "$VCF_FILE" ]; then
  echo "Error: VCF file '$VCF_FILE' does not exist."
  exit 1
fi

# create intermediate directory
LABEL_PREFIX="$OUTPUT_PREFIX/train"
mkdir -p "$LABEL_PREFIX"

generate_training_data() {
  local ctg_name=$1
  local start=$2
  local end=$3
  python -m errorprediction.generate_training_data \
    --confident_f $BED_FILE \
    --ref_f $REF_FILE \
    --label_out $LABEL_PREFIX/$ctg_name.$start.$end.label \
    --tagged_bam $BAM_FILE \
    --phased_vcf $VCF_FILE \
    --ctg_name $ctg_name \
    --ctg_start $start \
    --ctg_end $end \
    --window_size_half $WINDOW_SIZE_HALF

  if [ ! -s "$LABEL_PREFIX/$ctg_name.$start-$end.label" ]; then
    rm "$LABEL_PREFIX/$ctg_name.$start-$end.label"
  fi
}

export -f generate_training_data
export PARALLEL="--jobs $SLOTS"
export BAM_FILE REF_FILE BED_FILE VCF_FILE WINDOW_SIZE_HALF OUTPUT_PREFIX LABEL_PREFIX

# Generate chunks, only process regions from BED file if file is provided, otherwise for all ctgs
chunk_file="$LABEL_PREFIX/chunks.txt"
# touch "$chunk_file"

# Get chromosome names and lengths using samtools
#samtools idxstats "$BAM_FILE" | awk '{print $1, $2}' | while read -r chr length; do
#  for ((start = 1; start <= $length; start += CHUNK_SIZE)); do
#    end=$((start + CHUNK_SIZE - 1))
#    if [ $end -ge "$length" ]; then end=$length; fi
#    echo "$chr $start $end"
#  done
#done >"$chunk_file"

cat "$chunk_file" | parallel --colsep ' ' -j "$SLOTS" generate_training_data
