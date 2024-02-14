from collections import Counter
import itertools

# Given reads
rs = [('A', 0), ('A', 0), ('A', 0), ('A', 1), ('A', 0), ('A', 1), ('C', 1), ('C', 1), ('C', 1)]

# Error probability
error_prob = 0.003
# Probability of a specific incorrect nucleotide
specific_error_prob = error_prob / 3

# Possible genotypes
nucleotides = ['A', 'C', 'G', 'T']
possible_genotypes = [a + b for a, b in itertools.combinations_with_replacement(nucleotides, 2)]
print(possible_genotypes)
# Function to calculate likelihood of reads given a genotype
def calculate_likelihood(genotype, reads):
    likelihood = 1.0
    for read in reads:
        base, _ = read
        if base in genotype:
            # Correct read
            likelihood *= (1 - error_prob)
        else:
            # Incorrect read
            likelihood *= specific_error_prob
    return likelihood

# Calculate the likelihood for each genotype
genotype_likelihoods = {}
for genotype in possible_genotypes:
    genotype_likelihoods[genotype] = calculate_likelihood(genotype, rs)

# Find the genotype with the highest likelihood
most_likely_genotype = max(genotype_likelihoods, key=genotype_likelihoods.get)
most_likely_genotype, genotype_likelihoods[most_likely_genotype]

print(genotype_likelihoods, most_likely_genotype)