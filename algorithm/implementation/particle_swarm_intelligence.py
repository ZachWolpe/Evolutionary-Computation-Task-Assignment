#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Particle Swarm Optimization
@author: zachwolpe
WLPZAC001
"""

# Dependencies
import numpy as np
import pandas as pd

class particle:

    def __init__(self, knapsack, w, c1, c2, v_max, v_min, capacity):
        self.position = np.random.binomial(1, 0.5, 150)
        self.velocity = np.random.randint(v_min, v_max) * np.random.random()
        self.knapsack = knapsack
        self.capacity = capacity
        self.pbest = self.position
        self.pbest_score = 1
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_max
        self.v_min = v_min

    # ____________ ____________ ____________ Fix init Particles Evaluate Fitness ____________ ____________ ____________
    # ---- Evaluate Fitness
    def evaluate_genotype_fitness(self, genotype):
        """Evalute Fitness of Solution"""
        subset = self.knapsack.iloc[genotype == True,]
        if sum(subset['weight']) > self.capacity: fitness=1
        else: fitness=sum(subset['value'])
        return fitness

    def correct_initial_population(self, evaluation_function=evaluate_genotype_fitness):
        """As per Carsten's suggestion, randomly replace 1->0 in the initial population until accepted solutions achieved"""

        gene = self.position
        fitness = evaluation_function(self, gene)
        indices = [i_ for i_ in range(len(gene)) if gene[i_]==1]


        if len(indices)==0:
            gene = np.random.binomial(1, 0.5, 150)
            fitness = evaluation_function(self, gene)
            indices = [i_ for i_ in range(len(gene)) if gene[i_]==1]

        while fitness <= 1:
            c = np.random.choice(indices)   # select random index
            indices.remove(c)               # remove index
            gene[c]=0                       # 1->0
            fitness = evaluation_function(self, gene)

        self.position = np.array(gene)
        # return np.array(gene)







    def update_position(self, gbest, correct_initial_population=correct_initial_population):
        # returns a vector
        v1 = (self.w * self.velocity) + self.c1 * np.random.random() * (
                    self.pbest - self.position) + self.c2 * np.random.random() * (gbest - self.pbest)
        self.velocity = v1  # update velocity

        v1 = [self.v_max if v > self.v_max else v for v in v1]  # restrict max
        v1 = [self.v_min if v < self.v_min else v for v in v1]  # restrict max

        # binary: sigmoid transformation
        s1 = [1 / (1 + np.exp(-v)) for v in v1]  # probability vector
        s1 = np.array(s1) > np.random.random()  # E:{0,1} vector

        self.position = s1





    def update_pbest(self, fit_function):
        val = fit_function(self.position)  # evaluate current point
        if val > self.pbest_score:  # if better, update
            self.pbest = (self.position).astype(np.int)
            self.pbest_score = val

    def get_weight_value(self, best_known=997):
        """Return Weight & Value"""
        subset = self.knapsack.iloc[self.position == True,]
        weight = sum(subset['weight'])
        value = sum(subset['value'])
        results = {
            'genotype': self.position,
            'weight': weight,
            'value': value,
            'squality': value / best_known
        }
        return results






class PSO:

    def __init__(self, knapsack, specs, particle=particle, capacity=822, iterations=10000):
        self.particle = particle
        self.specs = specs
        self.capacity = capacity
        self.iterations = iterations

        min_vel = -int(specs['minimum_velocity'])
        max_vel = int(specs['maximum_velocity'])
        w = float(specs['inertia'])
        config = specs['configuration']
        n_particles = int(specs['number_particles'])
        c1 = float(specs['c1'])
        c2 = float(specs['c2'])



        # ---- Evaluate Fitness
        def evaluate_genotype_fitness(genotype):
            """Evalute Fitness of Solution"""
            subset = knapsack.iloc[genotype == True,]
            if sum(subset['weight']) > capacity: fitness=1
            else: fitness=sum(subset['value'])
            return fitness





        fit = pd.DataFrame(columns=['iteration', 'fitness', 'weight', 'value', 'squality', 'genotype'])                           # store data

        particles = [particle(knapsack=knapsack, w=w, c1=c1, c2=c2, v_min=min_vel, v_max=max_vel, capacity=capacity) for i in range(n_particles)]    # initiate


        # fix init particles
        [i.correct_initial_population() for i in particles]


        fitness = [evaluate_genotype_fitness(p.position) for p in particles]
        par = particles[fitness.index(max(fitness))]
        g_best = par.position                                                                                            # init gbest
        g_best_fit = evaluate_genotype_fitness(g_best)


        for i in range(iterations):

            # get fitness
            fitness = [evaluate_genotype_fitness(p.position) for p in particles]




            if max(fitness) > g_best_fit:

                par = particles[fitness.index(max(fitness))]           # find gbest
                g_best = (par.position).astype(np.int)                 # update gbest
                g_best_fit = max(fitness)                              # update gbest fit

            gwv = par.get_weight_value()  # update best results

            fit = fit.append(
                pd.DataFrame([[i] + [g_best_fit] + [gwv['weight']] + [gwv['value']] + [gwv['squality']] + [g_best]],
                             columns=['iteration', 'fitness', 'weight', 'value', 'squality', 'genotype']))

            # update position
            for p in particles:
                p.update_pbest(evaluate_genotype_fitness)
                p.update_position(g_best)
                # fix init particles
                p.correct_initial_population()


        # fix index
        fit.index = range(len(fit))

        # return results
        self.fit = fit



# import json
# file_name = 'pso_default_01'
# with open('data/json files/json_configuration_pso_default/' + file_name + '.json') as file:
#      specs = json.load(file)
#
#
#
# PSO(knapsack, specs)



# ----- for 1 particle -----x
# fitness = [evaluate_genotype_fitness(p.position) for p in particles]
#
# # find gbest
# par = particles[fitness.index(max(fitness))]
#
# if max(fitness) > g_best_fit:
#     g_best = (par.position).astype(np.int)  # update gbest
#     g_best_fit = max(fitness)  # update gbest fit
#
#
# for p in particles:
#     p.update_position(g_best)
#     p.update_pbest(evaluate_genotype_fitness)
#
#
#
#
#
# pp = particle(knapsack=knapsack, w=w, c1=c1, c2=c2, v_min=min_vel, v_max=max_vel, capacity=capacity)
# pp.position
# bestz = [130,450,690,780,880,910]
#
#
# f1 = evaluate_genotype_fitness(pp.position)
# pp.update_position(bestz[0])
#
# pp.position

