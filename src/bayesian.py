import itertools
import src.utils as utils

from collections import Counter


class GenotypeCaller:
    def __init__(self, reads, error_prob=0.003):
        self.reads = reads
        self.error_prob = error_prob
        self.specific_error_prob = error_prob / 3
        self.nucleotides = ["A", "C", "G", "T", "-"]
        self.possible_genotypes = [
            a + b
            for a, b in itertools.combinations_with_replacement(self.nucleotides, 2)
        ]
        self.genotype_likelihoods = {}

    def calculate_likelihood(self, genotype):
        likelihood = 1.0
        for read in self.reads:
            base, _ = read
            if base in genotype:
                likelihood *= 1 - self.error_prob
            else:
                likelihood *= self.specific_error_prob
        return likelihood

    def calculate_all_likelihoods(self):
        for genotype in self.possible_genotypes:
            self.genotype_likelihoods[genotype] = self.calculate_likelihood(genotype)

    def find_most_likely_genotype(self):
        self.calculate_all_likelihoods()
        most_likely_genotype = max(
            self.genotype_likelihoods, key=self.genotype_likelihoods.get
        )  # type: ignore
        return most_likely_genotype, self.genotype_likelihoods[most_likely_genotype]


class BayesianCaller:
    """
    Caller for one candidate site
    """

    def __init__(self, distributions):
        self.distributions = distributions
        self.cur_base_distribution = []  # from all reads
        self.next_base_distribution = []
        self.alleles = utils.ALLELES()

        for cur_b_dis, next_b_dis in distributions:
            self.cur_base_distribution.append(cur_b_dis)
            self.next_base_distribution.append(next_b_dis)

    def likelihood_per_read(alleles, observed_base, cur_b_d, next_b_d):
        """
        给定当前read的训练好的distributions, 计算在genotype下, 观察到observed_base的概率
        """
        return None

    def posterior_prob_of_genotype(self, genotype):
        likelihood = 1
        for i in range(self.__len__()):
            cur_b_dis = self.cur_base_distribution[i]
            next_b_dis = self.next_base_distribution[i]

            if genotype[0] in cur_b_dis and genotype[1] in next_b_dis:
                likelihood *= 1 - self.error_prob
            else:
                likelihood *= self.specific_error_prob

    def __len__(self):
        return len(self.cur_base_distribution)

    def __getitem__(self, idx):
        if idx < len(self.cur_base_distribution):
            return self.cur_base_distribution[idx], self.next_base_distribution[idx]
        else:
            return None, None


# Usage
# reads = [('A', 0), ('A', 0), ('A', 0), ('A', 1), ('A', 0), ('A', 1), ('C', 1), ('C', 1), ('C', 1)]
# caller = GenotypeCaller(reads)
# most_likely_genotype, likelihood = caller.find_most_likely_genotype()
# print(most_likely_genotype, likelihood)
