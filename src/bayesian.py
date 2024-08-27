import math
import src.utils as utils


class BayesianCaller:
    """
    Caller for one candidate site
    """

    def __init__(self):
        self.mini_prob = 1e-4
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

    def all_genotypes_posterior_prob_per_read_2(
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
            likelihood = 1e-4
            if key.startswith("snv"):
                # prior probability of each genotype under the observed_base
                prior_genotypes = (
                    self.base_prior_prob_dic[allele1]
                    * self.base_prior_prob_dic[allele2]
                )
                if observed_base in (allele1, allele2):
                    likelihood = self.base_prior_prob_dic[observed_base] / (
                        self.base_prior_prob_dic[allele1]
                        + self.base_prior_prob_dic[allele2]
                    )
                    # if allele1 == "A" and allele2 == "G":
                    #    print(
                    #        f"observe base: {observed_base}, allele1: {allele1}, allele2: {allele2}"
                    #    )
                    #    print(f"prior: {prior_genotypes}, likelihood: {likelihood}")
                likelihood = likelihood * prior_genotypes
            elif key.startswith("insertion"):
                prior_genotypes = (
                    self.base_prior_prob_dic[allele1]
                    * self.insertion_prior_prob_dic[allele2]
                )
                if observed_insertion == allele2:
                    likelihood = self.insertion_prior_prob_dic[observed_insertion]
                likelihood = likelihood * prior_genotypes

            pos_probs[key] = likelihood

        return pos_probs

    def all_genotypes_posterior_prob_per_read_2_log(
        self, cur_base_dis, insertion_dis, observed_base, observed_insertion
    ):
        # update base_prior_prob_dic and insertion_prior_prob_dic
        log_sum_base_dis, log_sum_ins_dis = self.process_and_sum_dis(
            cur_base_dis, insertion_dis
        )

        # calculate posterior probability for each genotype
        pos_probs = {}
        # print(self.base_prior_prob_dic, self.insertion_prior_prob_dic)

        for key, (allele1, allele2) in self.alleles.allele_dict.items():
            likelihood = self.mini_prob
            prior_genotypes = self.mini_prob

            log_observed_base = self.base_prior_prob_dic[observed_base]
            log_observed_ins = self.insertion_prior_prob_dic[observed_insertion]
            if key.startswith("snv"):
                prior_genotypes = (
                    self.base_prior_prob_dic[allele1]
                    + self.base_prior_prob_dic[allele2]
                )
                if observed_base in (allele1, allele2):
                    likelihood = math.exp(log_observed_base) / (
                        math.exp(self.base_prior_prob_dic[allele1])
                        + math.exp(self.base_prior_prob_dic[allele2])
                    )
                else:
                    likelihood = self.mini_prob
                    # total_error = math.exp(log_sum_base_dis) - math.exp(
                    #    self.base_prior_prob_dic[allele1]
                    # )
                    # if allele1 != allele2:
                    #    total_error -= math.exp(self.base_prior_prob_dic[allele2])

                    # likelihood = math.exp(log_observed_base) / total_error
                likelihood = math.log(likelihood)
                # print(
                #    likelihood,
                #    math.exp(log_observed_base),
                #    math.exp(log_sum_base_dis),
                #    math.exp(self.base_prior_prob_dic[allele1]),
                #    math.exp(self.base_prior_prob_dic[allele2]),
                # )

                # print(f"pre likelihood: {likelihood}")
                # print(f"after likelihood: {likelihood}")

            elif key.startswith("insertion"):
                prior_genotypes = (
                    self.insertion_prior_prob_dic[allele1]
                    + self.insertion_prior_prob_dic[allele2]
                )
                if observed_insertion in (allele1, allele2):
                    likelihood = math.exp(log_observed_ins) / (
                        math.exp(self.insertion_prior_prob_dic[allele1])
                        + math.exp(self.insertion_prior_prob_dic[allele2])
                    )
                else:
                    likelihood = self.mini_prob
                likelihood = math.log(likelihood)

            likelihood = likelihood + prior_genotypes
            # print(f"likelihood: {likelihood}")

            pos_probs[key] = likelihood

        return pos_probs

    def multiply_pos_probs_of_two_reads(self, pos_probs1, pos_probs2):
        """
        multiply two reads' posterior probs
        """
        pos_probs = {}
        for key, value in pos_probs1.items():
            pos_probs[key] = value * pos_probs2[key]
            # print(f"key: {key}, value: {value}, pos_probs2[key]: {pos_probs2[key]}")
        return pos_probs

    def add_pos_log_probs_of_two_reads(self, pos_log_probs1, pos_log_probs2):
        """
        add two reads' posterior log probs
        """
        pos_log_probs = {}
        for key, value in pos_log_probs1.items():
            pos_log_probs[key] = value + pos_log_probs2[key]
        return pos_log_probs

    def add_pos_probs_of_two_reads(self, pos_probs1, pos_probs2):
        """
        add two reads' posterior probs
        """
        pos_probs = {}
        for key, value in pos_probs1.items():
            pos_probs[key] = value + pos_probs2[key]
        return pos_probs

    def process_and_sum_dis(self, cur_base_dis, insertion_dis):
        sum_base_dis = 0
        sum_insertion_dis = 0
        for base, prob in zip(utils.CLASSES_PROB_1, cur_base_dis):
            if prob < self.mini_prob:
                sum_base_dis += self.mini_prob
                self.base_prior_prob_dic[base] = math.log(self.mini_prob)
            else:
                sum_base_dis += prob
                self.base_prior_prob_dic[base] = math.log(prob)

        for ins, prob in zip(utils.CLASSES_PROB_2, insertion_dis):
            if prob < self.mini_prob:
                sum_insertion_dis += self.mini_prob
                self.insertion_prior_prob_dic[ins] = math.log(self.mini_prob)
            else:
                sum_insertion_dis += prob
                self.insertion_prior_prob_dic[ins] = math.log(prob)

        log_sum_base_dis = math.log(sum_base_dis)
        log_sum_ins_dis = math.log(sum_insertion_dis)

        return log_sum_base_dis, log_sum_ins_dis

    def get_alleles(self):
        return self.alleles
