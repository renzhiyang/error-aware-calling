import math
import src.utils as utils


class BayesianCaller:
    """
    Caller for one candidate site
    """

    def __init__(self):
        self.mini_prob = 1e-10
        self.num_class_base = len(utils.CLASSES_PROB_1)
        self.num_class_ins = len(utils.CLASSES_PROB_2)
        self.alleles = utils.ALLELES()
        self.base_prior_prob_dic = {}
        self.insertion_prior_prob_dic = {}
        self.base_error_prob_dic = {}
        self.insertion_error_prob_dic = {}

        for base in utils.CLASSES_PROB_1:
            self.base_prior_prob_dic[base] = 0
        for ins in utils.CLASSES_PROB_2:
            self.insertion_prior_prob_dic[ins] = 0

    def __likelihood_allele(self, observed, allele, error_rate, is_snv):
        """
        Calculate the likelihood for allele under observed and error_rate
        """
        num_others = self.num_class_base - 1 if is_snv else self.num_class_ins - 1
        if observed == allele:
            return 1 - error_rate
        else:
            return error_rate / num_others

    def prior_probability_of_genotype_log(self, base_count, ins_count):
        """
        calculate prior probability of genotypes
        Input: an example base_count = {'-': 4, 'A': 3, 'C': 10, 'G': 20, 'T':2}
        ins_count is same as base_count
        """
        # Initialize allele frequency
        total_base = sum(base_count.values())
        total_ins = sum(ins_count.values())
        base_allele_freq = {
            base: count / total_base for base, count in base_count.items()
        }
        ins_allele_freq = {ins: count / total_ins for ins, count in ins_count.items()}

        prior_genotypes = {}
        for key, (allele1, allele2) in self.alleles.allele_dict.items():
            if key.startswith("snv"):
                prior_allele1 = base_allele_freq[allele1]
                prior_allele2 = base_allele_freq[allele2]
            else:
                prior_allele1 = ins_allele_freq[allele1]
                prior_allele2 = ins_allele_freq[allele2]

            # for homozygous and heterozygous
            if allele1 == allele2:
                prior_genotype = prior_allele1 * prior_allele2
            else:
                prior_genotype = 2 * prior_allele1 * prior_allele2

            if prior_genotype == 0:
                prior_genotype = self.mini_prob
            prior_genotype = math.log(prior_genotype)
            prior_genotypes[key] = prior_genotype
        return prior_genotypes

    def all_genotypes_likelihood_per_read_log(
        self, cur_base_dis, insertion_dis, observed_base, observed_insertion
    ):
        """
        calculate likelihood of observed base/insertion under genotypes
        """
        # Get the error rate of observed base
        base_error_rate = 1 - cur_base_dis[utils.CLASSES_PROB_1.index(observed_base)]
        ins_error_rate = (
            1 - insertion_dis[utils.CLASSES_PROB_2.index(observed_insertion)]
        )
        genotype_likelihoods = {}

        # calculate the likelihoods for one read
        for key, (allele1, allele2) in self.alleles.allele_dict.items():
            likelihood = self.mini_prob

            if key.startswith("snv"):
                likelihood_allele1 = self.__likelihood_allele(
                    observed_base, allele1, base_error_rate, True
                )
                likelihood_allele2 = self.__likelihood_allele(
                    observed_base, allele2, base_error_rate, True
                )
            else:
                likelihood_allele1 = self.__likelihood_allele(
                    observed_insertion, allele1, ins_error_rate, False
                )
                likelihood_allele2 = self.__likelihood_allele(
                    observed_insertion, allele2, ins_error_rate, False
                )
            # for homozygous and heterozygous
            if allele1 == allele2:
                likelihood = likelihood_allele1
            else:
                likelihood = 0.5 * likelihood_allele1 + 0.5 * likelihood_allele2

            likelihood = math.log(likelihood)
            genotype_likelihoods[key] = likelihood
        return genotype_likelihoods

    def all_genotypes_posterior_prob_per_read_log(
        self, cur_base_dis, insertion_dis, observed_base, observed_insertion
    ):
        # initialize base_prior_prob_dic and insertion_prior_prob_dic
        self.initialize_prior_prob_dic(cur_base_dis, insertion_dis)
        # print(
        # f"base_prior_prob_dic: {self.base_prior_prob_dic}"
        # )

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

        # marginal_likelihood = sum([1 / abs(value) for value in pos_probs.values()])
        # key, value in pos_probs.items():
        # pos_probs[key] = (1 / abs(value)) / marginal_likelihood
        # for key, value in pos_probs.items():
        # pos_probs[key] = value / marginal_likelihood

        return pos_probs

    def add_pos_probs_of_two_reads(self, pos_probs1, pos_probs2):
        """
        add two reads' posterior probs
        """
        pos_probs = {}
        for key, value in pos_probs1.items():
            pos_probs[key] = value + pos_probs2[key]
        return pos_probs

    def normalize_pos_probs_from_minus_input(self, pos_probs: dict):
        """
        normalize pos_probs from minus input
        """
        marginal_likelihood = sum(1 / abs(value) for value in pos_probs.values())
        for key, value in pos_probs.items():
            pos_probs[key] = (1 / abs(value)) / marginal_likelihood
        return pos_probs

    def initialize_prior_prob_dic(self, cur_base_dis, insertion_dis):
        """
        This method is used to initialize self.base_prior_prob_dic and self.insertion_prior_prob_dic
        Let probabilities be quality score, so we can get the error rates
        Input:
            cur_base_dis: a 1x5 list, restore the probabilities of A,C,G,T,- predicted by DL model
            insertion_dis: a 1x25 list, restore the probabilities of insertions predicted by DL model
        """
        for base, prob in zip(utils.CLASSES_PROB_1, cur_base_dis):
            if prob < self.mini_prob:
                self.base_prior_prob_dic[base] = math.log(self.mini_prob)
                self.base_error_prob_dic[base] = 1 - self.mini_prob
            else:
                self.base_prior_prob_dic[base] = math.log(prob)
                self.base_error_prob_dic[base] = 1 - prob

        for ins, prob in zip(utils.CLASSES_PROB_2, insertion_dis):
            if prob < self.mini_prob:
                self.insertion_prior_prob_dic[ins] = math.log(self.mini_prob)
                self.base_error_prob_dic[ins] = 1 - self.mini_prob
            else:
                self.insertion_prior_prob_dic[ins] = math.log(prob)
                self.base_error_prob_dic[ins] = 1 - prob

    def get_alleles(self):
        return self.alleles
