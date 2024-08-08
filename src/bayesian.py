import src.utils as utils


class BayesianCaller:
    """
    Caller for one candidate site
    """

    def __init__(self):
        self.alleles = utils.ALLELES()
        self.base_prior_prob_dic = {}
        self.insertion_prior_prob_dic = {}

        for base in utils.CLASSES_PROB_1:
            self.base_prior_prob_dic[base] = 0
        for ins in utils.CLASSES_PROB_2:
            self.insertion_prior_prob_dic[ins] = 0

    def all_genotypes_posterior_porb_per_read(
        self, cur_base_dis, insertion_dis, observed_base, observed_insertion
    ):
        """
        给定当前read的训练好的distributions, 计算在genotype下, 观察到observed_base的概率
        输入：当前read的distributions,所有的alleles
        输出：给出在cur_base_dis,insertion_dis的前提下，所有genotype的后验概率
        """
        # update base_prior_prob_dic and insertion_prior_prob_dic
        for base, prob in zip(utils.CLASSES_PROB_1, cur_base_dis):
            self.base_prior_prob_dic[base] = prob
        for ins, prob in zip(utils.CLASSES_PROB_2, insertion_dis):
            self.insertion_prior_prob_dic[ins] = prob

        # calculate posterior probability for each genotype
        pos_probs = {}
        for key, (allele1, allele2) in self.alleles.allele_dict.items():
            # 1. calculate likelihood
            # 2. posterior = likelihood * prior genotype

            # calculate likelihood
            likelihood = 0
            if key.startswith("snv"):
                likelihood = (
                    self.base_prior_prob_dic[allele1]
                    * self.base_prior_prob_dic[allele2]
                )
            elif key.startswith("insertion"):
                likelihood = (
                    self.base_prior_prob_dic[allele1]
                    * self.insertion_prior_prob_dic[allele2]
                )

            likelihood *= (
                self.base_prior_prob_dic[observed_base]
                * self.insertion_prior_prob_dic[observed_insertion]
            )
            pos_probs[key] = likelihood

        return pos_probs

    def multiply_pos_probs_of_two_reads(self, pos_probs1, pos_probs2):
        """
        multiply two reads' posterior probs
        """
        pos_probs = {}
        for key, value in pos_probs1.items():
            pos_probs[key] = value * pos_probs2[key]
        return pos_probs

    def get_alleles(self):
        return self.alleles


# Usage
# reads = [('A', 0), ('A', 0), ('A', 0), ('A', 1), ('A', 0), ('A', 1), ('C', 1), ('C', 1), ('C', 1)]
# caller = GenotypeCaller(reads)
# most_likely_genotype, likelihood = caller.find_most_likely_genotype()
# print(most_likely_genotype, likelihood)
