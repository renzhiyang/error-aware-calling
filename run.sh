#!/bin/bash

usage() {
  echo "Usage: ./run.sh -b <bam_file> -r <ref_file> -o <output_prefix> -t <slots>"
  echo "Example: ./run.sh -b input.bam -r reference.bam -o output_pileup -t 4"
  echo 'Required parameters:'
  echo '  -b, --bam_fn=FILE            BAM file input. The input file must be samtools indexed.'
  echo '  -r, --ref_fn=FILE            Reference file input.'
  echo '  -o, --output_prefix=FILE     Output file prefix.'
  echo '  -t, --slots=INT              Number of slots to use.'
  echo '  -m, --model=FILE             Model file input.'
  echo ''
  echo 'Optional parameters:'
  echo '  --bed_fn=FILE                The regions that will be processed.'
  echo '  --config_f=FILE              The configuration file.'
  echo '  --chunck_size=INT            The chunk size for parallel processing.'
  echo '  --samtools=STR               Path to samtools.'
  exit 1
}

version() {
  echo "run.sh version 1.0"
}

# Check if no arguments are passed
if [ $# -eq 0 ]; then
  usage
fi

# Default values
BED_FILE=""
CHUNK_SIZE=1000000
SAMTOOLS="samtools"

############### Parse command line arguments
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
  -m | --model)
    MODEL="$2"
    shift
    ;;
  --config_f)
    CONFIG_FILE="$2"
    shift
    ;;
  --bed_fn)
    BED_FILE="$2"
    shift
    ;;
  --chunck_size)
    CHUNK_SIZE="$2"
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
if [ -z "$MODEL" ] || [ -z "$BAM_FILE" ] || [ -z "$REF_FILE" ] || [ -z "$OUTPUT_PREFIX" ] || [ -z "$SLOTS" ]; then
  echo "Error: Missing required parameters."
  usage
fi

############### Ensure output directory exists
mkdir -p "$OUTPUT_PREFIX"

############### Ensure required software is installed
required_software=("samtools" "parallel" "python")

for software in "${required_software[@]}"; do
  if ! command -v "$software" &>/dev/null; then
    echo "Error: $software is not installed."
    exit 1
  fi
done

############### Confirm the inputs
echo "Input Parameters:"
echo "Model File: $MODEL"
echo "BAM File: $BAM_FILE"
echo "Reference File: $REF_FILE"
echo "Output Prefix: $OUTPUT_PREFIX"
echo "Slots: $SLOTS"
echo "BED File: $BED_FILE"
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

# create intermediate directory
CANDIDATES_PREFIX="$OUTPUT_PREFIX/candidates"
TENSOR_PREFIX="$OUTPUT_PREFIX/tensor"
VCF_PREFIX="$OUTPUT_PREFIX/vcf"
mkdir -p "$CANDIDATES_PREFIX"
mkdir -p "$TENSOR_PREFIX"
mkdir -p "$VCF_PREFIX"

############### FIRST STEP: Find candidates and generate tensor file
find_candidates_generate_tensors() {
  local ctg_name=$1
  local ctg_start=$2
  local ctg_end=$3
  # echo "$CANDIDATES_PREFIX/$ctg_name.$ctg_start-$ctg_end.candidates"
  python ./src/find_candidates.py \
    --bam_fn "$BAM_FILE" \
    --ref_fn "$REF_FILE" \
    --ctg_name "$ctg_name" \
    --ctg_start "$ctg_start" \
    --ctg_end "$ctg_end" \
    --min_mq 20 \
    --min_coverage 10 \
    --min_allele_freq 0.125 \
    >"$CANDIDATES_PREFIX/$ctg_name.$ctg_start-$ctg_end.candidates"
  # Remove file if it's empty
  if [ ! -s "$CANDIDATES_PREFIX/$ctg_name.$ctg_start-$ctg_end.candidates" ]; then
    rm "$CANDIDATES_PREFIX/$ctg_name.$ctg_start-$ctg_end.candidates"
  else
    echo "genrate tensor at $ctg_name:$ctg_start-$ctg_end"
    python -m src.generate_tensor \
      --ref_fn "$REF_FILE" \
      --bam_fn "$BAM_FILE" \
      --candidates_fn "$CANDIDATES_PREFIX/$ctg_name.$ctg_start-$ctg_end.candidates" \
      --tensor_fn "$TENSOR_PREFIX/$ctg_name.$ctg_start-$ctg_end.tensor"
  fi
}

export -f find_candidates_generate_tensors
export PARALLEL="--jobs $SLOTS"
export BAM_FILE REF_FILE OUTPUT_PREFIX CANDIDATES_PREFIX TENSOR_PREFIX VCF_PREFIX

# Function to combine candidate files for each chromosome
combine_candidates() {
  local ctg_name=$1
  cat "$OUTPUT_PREFIX/$ctg_name.*.candidates" >"$OUTPUT_PREFIX/$ctg_name.combined.candidates"
  # Remove intermediate files
  rm "$OUTPUT_PREFIX/$ctg_name.*.candidates"
}
export -f combine_candidates

# Generate chunks, only process regions from BED file if file is provided, otherwise for all ctgs
chunk_file="$OUTPUT_PREFIX/chunks.txt"
touch "$chunk_file"

# generate chunk file, each line is a region to process
if [ -n "$BED_FILE" ]; then
  # Process regions from BED file
  awk -v chunk_size="$CHUNK_SIZE" '{
        for (start = $2 + 1; start <= $3; start += chunk_size) {
            end = start + chunk_size - 1
            if (end > $3) end = $3
            printf "%s %d %d\n", $1, start, end
        }
    }' "$BED_FILE" >"$chunk_file"
else
  # Get chromosome names and lengths using samtools
  samtools idxstats "$BAM_FILE" | awk '{print $1, $2}' | while read -r chr length; do
    for ((start = 1; start <= $length; start += CHUNK_SIZE)); do
      end=$((start + CHUNK_SIZE - 1))
      if [ $end -ge "$length" ]; then end=$length; fi
      echo "$chr $start $end"
    done
  done >"$chunk_file"
fi

# Process chunks using GNU parallel with limited threads
# cat "$chunk_file" | parallel --colsep ' ' -j "$SLOTS" find_candidates_generate_tensors

# delete candidate files
# rm "$CANDIDATES_PREFIX"/*.candidates

############### Step2: Predicting ###############
predict() {
  local tensor_f=$1
  vcf_filename="${tensor_f##*/}"
  vcf_name="${vcf_filename%.tensor}"
  python predict.py \
    --model "$MODEL" \
    --tensor_fn $tensor_f \
    --output_fn "$VCF_PREFIX/$vcf_name.vcf" \
    --config_fn "$CONFIG_FILE"
}

export MODEL CONFIG_FILE
export -f predict

# process tensor files using GNU parallel with limited threads
find "$TENSOR_PREFIX" -name "*.tensor" | parallel --jobs "$SLOTS" predict
