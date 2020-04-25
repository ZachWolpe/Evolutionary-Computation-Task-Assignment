#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:07:56 2020

Genetic Algorithms
@author: zachwolpe
WLPZAC001
"""

# Genetic Algorithms Task Assignment
import numpy as np
import pandas as pd



class knapsack_genetic_algorithms:
    """
    Instantiate:                  All functions to compute genetic algorithm
    
    Arguments: 
        knapsack:            problem set
        pop_size:            population size
        configuration:       configuration name
        selection:           {'RWS', 'TS'} parent selection method
        crossover:           {'1PX', '2PX'} crossover method
        crossover_p:         {0.6, 0.7, 0.8} probability of crossover
        mutation:            {'BFM', 'EXM', 'IVM', 'ISM', 'DPM'} mutation method
        mutation_p:          {0.003} probability of mutation
        capacity:            the contraint on the knapsack problem
        max_iters:           maximum iterations (if not converged)
    """

    def __init__(self, knapsack, pop_size, capacity=822, max_iters=10000):

        self.knapsack = knapsack
        self.pop_size = pop_size
        self.capacity = capacity
        self.max_iters = max_iters
        self.init_population = np.array([np.random.binomial(1, 0.5, 150) for i in range(pop_size)])
        # np.array([[random.randint(0,1) for i in range(150)] for i in range(pop_size)])
        # self.init_population = np.reshape([random.randint(0,1) for i in range(150*ks.pop_size)], (ks.pop_size, 150))

    # ____________ ____________ ____________ Evaluate Fitness ____________ ____________ ____________
    def evaluate_genotypes_fitness(self, genotypes):
        """Evalute Fitness of Solution"""

        fitness = []
        for g in genotypes:

            subset = self.knapsack.iloc[g == True,]

            if sum(subset['weight']) > self.capacity:
                fitness.append(1)
            else:
                fitness.append(sum(subset['value']))

        return fitness

    def evaluation_function(self, genotype):
        """Evaluation for a single genotype"""
        subset = self.knapsack.iloc[genotype==True,]
        if sum(subset['weight']) > self.capacity: return 1
        else: return sum(subset['value'])


    def correct_initial_population(self, population, evaluation_function=evaluation_function):
        """As per Carsten's suggestion, randomly replace 1->0 in the initial population until accepted solutions achieved"""
        genes = []
        for gene in population:
            fitness = evaluation_function(self, gene)
            indices = [i_ for i_ in range(len(gene)) if gene[i_]==1]
            while fitness <= 1:

                c = np.random.choice(indices)   # select random index
                indices.remove(c)               # remove index
                gene[c]=0                       # 1->0

                fitness = evaluation_function(self, gene)
            genes.append(gene)

        return np.array(genes)




    # ____________ ____________ ____________ Evaluate Fitness ____________ ____________ ____________

    def get_weight_value(self, genotype):
        """Return Weight & Value"""
        subset = self.knapsack.iloc[genotype == True,]
        weight = sum(subset['weight'])
        value = sum(subset['value'])
        results = {
            'genotype': genotype,
            'weight': weight,
            'value': value
        }
        return results

    # ____________ ____________ ____________ Parent Selection ____________ ____________ ____________
    def RWS(self, population, fitness_scores):
        """Rolette Wheel Selection - return 1 parent"""

        prob_select = np.array(fitness_scores) / sum(np.array(fitness_scores))  # probability of selection
        i = np.random.choice(population.shape[0], size=1, p=prob_select)  # select - probabilistic sampling is equivalent to a wheel

        return population[i]

    def tournament_selection(self, population, tournament_size, fitness_scores):
        """deterministic tounrament parent selection - return 1 parent"""

        i = np.random.choice(population.shape[0], tournament_size)

        # parents 
        parents = population[i]

        # evaluate fitness
        if (tournament_size == 1):
            f = fitness_scores[i]
        else:
            f = [fitness_scores[a] for a in i]

        # takes first configuration with highest value, which should be fine
        f = f.index(max(f))
        return (np.array(parents[f]))

    # ____________ ____________ ____________ Parent Selection ____________ ____________ ____________

    # ____________ ____________ ____________ Recombination ____________ ____________ ____________
    def crossover(self, parent1, parent2, px='1PX'):
        """Return 2 childern after crossover"""

        if px == '1PX':
            point = np.random.choice(range(150 - 1))
            child1 = np.append(parent1[:point], parent2[point:])
            child2 = np.append(parent2[:point], parent1[point:])

        if px == '2PX':
            point1 = np.random.choice(range(150 - 1))
            if point1 == (150 - 1):
                point2 = point1
            else:
                point2 = np.random.choice(range(point1, 150 - 1))

            child1 = np.append(
                np.append(parent1[:point1], parent2[point1:point2]), parent1[point2:])
            child2 = np.append(
                np.append(parent2[:point1], parent1[point1:point2]), parent2[point2:])

        return (np.array([child1, child2]))

    # ____________ ____________ ____________ Recombination ____________ ____________ ____________

    # ____________ ____________ ____________ ____________ Mutation ____________ ____________ ____________ ____________
    def bit_flip_mutation(self, gen_code):
        """Bit Flip Mutation"""

        # select indices
        i = np.random.choice(len(gen_code), 1)

        if gen_code[i] == 1:
            gen_code[i] = 0
        else:
            gen_code[i] = 1

        return (gen_code)

    def exchange_mutation(self, gen_code):
        """Exchange Mutation"""

        # select indices
        i = np.random.choice(len(gen_code), 2)

        x = gen_code[i[0]]
        gen_code[i[0]] = gen_code[i[1]]
        gen_code[i[1]] = x

        return (gen_code)

    def inverse_mutation(self, gen_code):
        """Inverse Mutation"""

        # select indices
        i = np.random.choice(len(gen_code), 2)

        # invert sequence between two indices
        if i[0] != i[1]:
            gen_code[min(i):max(i)] = np.flip(gen_code[min(i):max(i)])
            return (gen_code)

        else:
            return (gen_code)

    def insert_mutation(self, gen_code):
        """Insert Mutation"""

        # select indices
        i = np.random.choice(len(gen_code), 2)

        # invert sequence between two indices
        if i[0] != i[1]:

            # move segment to middle
            gen_code = np.append(gen_code[:min(i)],
                                 np.append(gen_code[max(i):], gen_code[min(i):max(i)]))
            return (gen_code)

        else:
            return (gen_code)

    def displacement_mutation(self, gen_code):
        """Displacement Mutation"""

        # select indices
        i = np.random.choice(len(gen_code), 2)

        # invert sequence between two indices
        if i[0] != i[1]:

            strip = gen_code[min(i):max(i)]  # select strip
            s = np.append(gen_code[:min(i)], gen_code[max(i):])  # other section (not selected)

            # select random insert location
            l = np.random.choice(len(gen_code))

            # insert at location
            gen_code = np.append(np.append(s[:l], strip), s[l:])

            return (gen_code)
        else:
            return (gen_code)
            # ____________ ____________ ____________ ____________ Mutation ____________ ____________ ____________ ____________




def compute_genetic_algorithm(knapsack, specs, iterations=10000, knapsack_genetic_algorithms=knapsack_genetic_algorithms):
    ks = knapsack_genetic_algorithms(knapsack, capacity=822, pop_size=100)

    # init population
    pop = ks.init_population

    # correct init population (Carsten update)
    pop = ks.correct_initial_population(ks.init_population)

    # store results
    fit = pd.DataFrame(columns=['iteration', 'fitness', 'weight', 'value', 'squality', 'genotype'])
    count = 0

    for i in range(iterations):

        # evaluate fitness
        fitness = ks.evaluate_genotypes_fitness(pop)
        # keep best fitness per generation
        ge = ks.get_weight_value(pop[fitness.index(max(fitness))])
        val = [ge['value'] if ge['weight'] < ks.capacity else np.nan for i in range(1)]
        fit.loc[i] = [i] + [max(fitness)] + [ge['weight']] + [val[0]] + [val[0] / 997] + [
            pop[fitness.index(max(fitness))]]

        # --- - --- - --- Generation --- - --- - ---
        children = []
        for j in range(int(pop.shape[0] / 2)):

            # select parents
            if specs['selection_method'] == 'RWS':
                p1 = ks.RWS(pop, fitness)
                p2 = ks.RWS(pop, fitness)

            elif specs['selection_method'] == 'TS':
                p1 = ks.tournament_selection(pop, 3, fitness)
                p2 = ks.tournament_selection(pop, 3, fitness)

            # recombination
            if np.random.uniform() < float(specs['crossover_ratio']):
                children_ = ks.crossover(p1, p2, specs['crossover_method'])
            else:
                children_ = np.array([p1, p2]).reshape((2, 150))

            # mutation
            if np.random.uniform() < float(specs['mutation_ratio']):

                if specs['mutation_method'] == 'BFM':
                    c1 = ks.bit_flip_mutation(children_[0])
                    c2 = ks.bit_flip_mutation(children_[1])

                if specs['mutation_method'] == 'EXM':
                    c1 = ks.exchange_mutation(children_[0])
                    c2 = ks.exchange_mutation(children_[1])

                if specs['mutation_method'] == 'IVM':
                    c1 = ks.inverse_mutation(children_[0])
                    c2 = ks.inverse_mutation(children_[1])

                if specs['mutation_method'] == 'ISM':
                    c1 = ks.insert_mutation(children_[0])
                    c2 = ks.insert_mutation(children_[1])

                if specs['mutation_method'] == 'DPM':
                    c1 = ks.displacement_mutation(children_[0])
                    c2 = ks.displacement_mutation(children_[1])

                children_ = np.array([c1, c2])

            children.append(children_)  # store results
        children = np.array(children).reshape(100, 150)  # reshape

        # Keep best Parents vs Children
        candidates = np.append(children, pop, axis=0)
        ff = ks.evaluate_genotypes_fitness(candidates)
        candidates = candidates[np.argsort(ff)[-100:]]

        # select best
        pop = candidates
        # --- - --- - --- Generation --- - --- - ---

    return fit

# ____________ ____________ ____________ ____________ Algorithm ____________ ____________ ____________ ____________










# Knapsack Problem

# In the 01 Knapsack problem, we are given a knapsack of fixed capacity C. 
# We are also given a list of N objects, each having a 
# weight W(I) and profit P(I). We can put any subset of the objects into 
# the knapsack, as long as the total weight of our selection does not exceed C.
# We desire to maximize our total profit, which is the sum of the profits of 
# each object we put into the knapsack.

