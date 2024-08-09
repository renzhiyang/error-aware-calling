import pysam

phased_vcf = (
    "/home/yang1031/projects/error-aware-calling/data/hg001_chr21_phased.vcf.gz"
)
out_tsv = "/home/yang1031/projects/error-aware-calling/data/hg001_chr21_phased.tsv"
# Open the phased VCF file
vcf = pysam.VariantFile(phased_vcf)

# Open the output TSV file for writing
with open(out_tsv, "w") as output:
    # Write the header
    output.write("#CHROM\tPOS\tID\tREF\tALT\tHAPLOTYPE_1\tHAPLOTYPE_2\n")

    # Iterate over each record in the VCF
    for record in vcf:
        # Extract basic variant information
        chrom = record.chrom
        pos = record.pos
        var_id = record.id if record.id else "."
        ref = record.ref
        alt = ",".join(str(a) for a in record.alts)

        # Extract the phased genotype information (assuming a single sample VCF)
        # This will look like (0|1) for heterozygous variants
        genotype = record.samples[0]["GT"]

        # Determine the haplotype assignments
        haplotype_1 = genotype[0]
        haplotype_2 = genotype[1] if len(genotype) > 1 else "."

        # Write the variant and its haplotype information to the output file
        output.write(
            f"{chrom}\t{pos}\t{var_id}\t{ref}\t{alt}\t{haplotype_1}\t{haplotype_2}\n"
        )
