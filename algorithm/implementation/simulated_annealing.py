"""
Genetic Algorithms
@author: zachwolpe
WLPZAC001
"""

# Genetic Algorithms Task Assignment
import numpy as np
import pandas as pd
import random
import json





class knapsack_sa_algorithms:
    """
    Return:                  All functions to compute genetic algorithm

    Arguments:
        knapsack:            problem set
        pop_size:            population size
        capacity:            the contraint on the knapsack problem
        max_iters:           maximum iterations (if not converged)
    """

    def __init__(self, knapsack, capacity=822):

        self.knapsack = knapsack
        self.capacity = capacity
        self.init_population = np.random.binomial(1, 0.5, 150)

    # ____________ ____________ ____________ Evaluate Fitness ____________ ____________ ____________
    def evaluate_genotypes_fitness(self, genotype):
        """Evalute Fitness of Solution"""

        subset = self.knapsack.iloc[genotype == True,]
        if sum(subset['weight']) > self.capacity: fitness = 1
        else: fitness = sum(subset['value'])
        return fitness


    def correct_initial_population(self, gene, evaluation_function=evaluate_genotypes_fitness):
        """As per Carsten's suggestion, randomly replace 1->0 in the initial population until accepted solutions achieved"""

        fitness = evaluation_function(self, gene)
        indices = [i_ for i_ in range(len(gene)) if gene[i_]==1]

        while fitness <= 1:
            c = np.random.choice(indices)   # select random index
            indices.remove(c)               # remove index
            gene[c]=0                       # 1->0
            fitness = evaluation_function(self, gene)
        return np.array(gene)



    # ____________ ____________ ____________ Evaluate Fitness ____________ ____________ ____________

    def get_weight_value(self, gene, best_known=997, capacity=822):
        """Return Weight & Value"""
        subset = self.knapsack.iloc[gene == True,]
        weight = sum(subset['weight'])
        value = sum(subset['value'])
        if weight > capacity:
            value = np.NaN
        results = {
            'genotype': gene,
            'weight': weight,
            'value': value,
            'squality': value / best_known
        }
        return results







class compute_sa_algorithm:

    def __init__(self, knapsack, specs, capacity=822, knapsack_sa_algorithms=knapsack_sa_algorithms):


        ks = knapsack_sa_algorithms(knapsack=knapsack, capacity=capacity)
        # ks = knapsack_sa_algorithms(knapsack=knapsack, capacity=822)

        s = ks.init_population

        # correct init population
        s = ks.correct_initial_population(s)



        cooling_rate = (1 - float(specs['cooling_rate'])/2000)
        fitness = []

        Temp = float(specs['initial_temperature'])
        fit = pd.DataFrame(columns=['iteration', 'fitness', 'weight', 'value', 'squality', 'genotype'])

        j = 0
        while round(Temp) > 0:
            # generate neighbour solution?
            s1 = np.random.binomial(1, 0.5, 150)
            s1 = ks.correct_initial_population(s1)

            # evaluate
            E = ks.evaluate_genotypes_fitness(s1) - ks.evaluate_genotypes_fitness(s)

            if E < 0:
                s = s1
            elif np.exp(E / Temp) > random.random():
                s = s1

            # store results
            f = ks.evaluate_genotypes_fitness(s)
            fitness.append(f)

            gwv = ks.get_weight_value(s)
            fit = fit.append(
                pd.DataFrame([[j] + [f] + [gwv['weight']] + [gwv['value']] + [gwv['squality']] + [s]],
                             columns=['iteration', 'fitness', 'weight', 'value', 'squality', 'genotype']))

            Temp = cooling_rate * Temp
            j += 1

        fit.index = range(len(fit))
        self.fitness = fit














        #
        #
        # for i in range(iterations):
        #     # use best metastable solution each time
        #     if i > 0:
        #         s = c_best['genotype']
        #
        #     fitness = []
        #     Temp = float(specs['initial_temperature'])
        #     fit = pd.DataFrame(columns=['iteration', 'fitness', 'weight', 'value', 'squality', 'genotype'])
        #     j = 0
        #
        #     while round(Temp) > 0:
        #         # generate neighbour solution?
        #         s1 = np.random.binomial(1, 0.5, 150)
        #         s1 = ks.correct_initial_population(s1)
        #
        #         # evaluate
        #         E = ks.evaluate_genotypes_fitness(s1) - ks.evaluate_genotypes_fitness(s)
        #
        #         if E < 0:
        #             s = s1
        #         elif np.exp(E / Temp) > random.random():
        #             s = s1
        #
        #         # store results
        #         f = ks.evaluate_genotypes_fitness(s)
        #         fitness.append(f)
        #
        #         gwv = ks.get_weight_value(s)
        #         fit = fit.append(
        #             pd.DataFrame([[j] + [f] + [gwv['weight']] + [gwv['value']] + [gwv['squality']] + [s]],
        #                          columns=['iteration', 'fitness', 'weight', 'value', 'squality', 'genotype']))
        #
        #         Temp = cooling_rate * Temp
        #         j += 1
        #
        #     fit.index = range(len(fit))
        #     # save best as starting point
        #
        #
        #
        #     c_best = fit.iloc[pd.Index(fit.fitness).get_loc(max(fit.fitness)), ]
        #     if c_best.shape != (6,):
        #         c_best = c_best.iloc[0, ]
        #
        #     fit_final = fit_final.append(
        #         pd.DataFrame([[i] + [c_best['fitness']] + [c_best['weight']] + [c_best['value']] + [c_best['squality']] + [c_best['genotype']]],
        #                      columns=['iteration', 'fitness', 'weight', 'value', 'squality', 'genotype']))
        #
        #
        #
        #
        # fit_final.index = range(len(fit_final))
