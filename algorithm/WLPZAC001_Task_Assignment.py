# Dependencies
import sys
import json
import numpy as np
import pandas as pd
import timeit
import os
from tqdm import tqdm
from datetime import datetime

from implementation import genetic_algorithm
from implementation import particle_swarm_intelligence
from implementation import simulated_annealing

knapsack = pd.read_csv('data/knapsack_instance.csv', sep=';')




def generate_report(specs, fit, best_solution, subclass, compute_time, one_report=True):
    # generate time
    now = datetime.now()

    # slice
    fitr = fit[['weight', 'value', 'squality', 'genotype']]

    # create quantile summary
    summary = fitr.iloc[[int(len(fitr) / 4 * 1) - 1, int(len(fitr) / 4 * 2) - 1, int(len(fitr) / 4 * 3) - 1,
                         int(len(fitr) / 4 * 4) - 1], :3]
    summary.index = summary.index + 1

    # Calculate Longest Plateau
    diff = 0
    for i in range(1, len(fitr['squality'])):
        for j in range(i + 1, len(fitr['squality'])):
            if fitr['squality'][i] * 1.03 < fitr['squality'][j]:
                break
        if (j - i) > diff:
            x, x_ = i, j
            diff = j - i
    plateau = 'Longest sequence, {}-{} with improvement less average 3%'.format(x, x_)


    if subclass == 'ga':
        conf = 'GA | #{} | {} | {} ({}) | {} ({})'.format(len(fit), specs['selection_method'], specs['crossover_method'],
                                                       specs['mutation_ratio'], specs['mutation_method'], specs['crossover_ratio'])

    if subclass == 'pso':
        conf = '{} | #{} | inertia={} | n particles={} | c1={} | c2={}'.format(subclass, len(fit), specs['inertia'],
            specs['number_particles'], specs['c1'], specs['c2'])

    if subclass == 'sa':
        conf = '{} | #{} | Initial Temp:{} | Cooling Rate:{}'.format(subclass, len(fit), specs['initial_temperature'],
                                                               specs['cooling_rate'])


    # get best run
    # best_loc = pd.Index(fitr['squality']).get_loc(max(fitr['squality']))
    # br = fitr.iloc[best_loc, ]
    # if br.shape != (4,):
    #     br = br.iloc[0,]
    br = best_solution


    print('- - - - - - - - - - - -')
    print(br)





    if one_report:
        temp_file = open('reports/specific_reports/report_{}_{}.txt'.format(subclass, specs['configuration'], now.strftime("%Y%m%d")), 'w')

    else:
        temp_file = open('reports/config/{}/report_{}_{}.txt'.format(subclass, specs['configuration'], now.strftime("%Y%m%d")), 'w')


    message = '''
    
    Evaluation | {}
    Configuration:  {}.json
                    
                    {}

=================================================================================
    
    {}
    
---------------------------------------------------------------------------------
    
    [Statistics]
    
    Runtime             {} ms
    
    Convergence  
           
    {}
    
    Plateau
    
    {}
    
    Best Run
    
            index:      {} 
            weight:     {}
            value:      {} 
            squality:   {}
            genotype:   {}{}{}{}{}{}{}{}{}{}...    
    
=================================================================================

    '''.format(now.strftime("%Y-%m-%d %H:%M"), specs['configuration'], conf, fitr, compute_time, summary, plateau, br.name,
               br['weight'], br['value'], br['squality'], *br['genotype'])
    temp_file.write(message)
    temp_file.close()
    print(message)



def run_configuration(file_name, knapsack, subclass, iters=10000, gen_report=True, one_report=True, generate_report=generate_report):
    start = timeit.default_timer()

    if subclass == 'ga':
        with open('data/json files/json_configuration_ga_default/' + file_name + '.json') as file:
            specs = json.load(file)
        ga = genetic_algorithm.compute_genetic_algorithm(knapsack, specs, iterations=iters)
        fit = ga

    elif subclass == 'pso':
        with open('data/json files/json_configuration_pso_default/' + file_name + '.json') as file:
            specs = json.load(file)
        pso = particle_swarm_intelligence.PSO(knapsack, specs, iterations=1000)
        fit = pso.fit

    elif subclass == 'sa':
        with open('data/json files/json_configuration_sa_default/' + file_name + '.json') as file:
            specs = json.load(file)
        sa = simulated_annealing.compute_sa_algorithm(knapsack, specs)
        fit = sa.fitness

    # compute time
    stop = timeit.default_timer()
    ms = round((stop - start) * 1000, 4)

    best_loc = pd.Index(fit['squality']).get_loc(max(fit['squality']))
    br = fit.iloc[best_loc, ]
    if br.shape != (6,):
        br = br.iloc[0, ]

    if gen_report:
        generate_report(specs, fit, br, subclass, ms, one_report)


    results = {
        'file name': file_name,
        'fit': fit,
        'best solution': br,
        'miliseconds': ms
    }
    return results



def write_best_config(sub):
    """Search the current configurations & locate & write the best to 'best_config'"""

    paths = os.listdir('reports/config/' + sub)
    paths = ['reports/config/' + sub + '/' + i for i in paths]
    best_squality = 0

    for i in paths:
        if '.DS_Store' not in i:
            with open(i) as f:
                last_squality = ''
                for position, line in enumerate(f):
                    if 'squality' in line:
                        last_squality = line
                squality = float(last_squality.split()[1])

            if squality >= best_squality:
                best_squality = squality
                best_file_path = i

            f.close()

    with open(best_file_path) as bf:
        tempfile = open('reports/best_config/{}/{}.txt'.format(sub, best_file_path.split('/')[-1].split('.')[0]), 'w')
        tempfile.write(bf.read())
        tempfile.close()



def generate_all_reports(typ=['ga', 'pso', 'sa'], ga_iters=10000, pso_iters=1000):

    all_configs = pd.DataFrame(columns=['configuration', 'weight', 'value', 'squality', 'genotype'])

    for t in typ:

        if t == 'ga':
            print('____ starting GA ____')
            for filename in tqdm(os.listdir('data/json files/json_configuration_ga_default/')):
                f = os.path.splitext(filename)[0]
                xx = run_configuration(f, knapsack, t, iters=ga_iters)
                all_configs = all_configs.append(
                    pd.DataFrame(
                        data=[[filename, *xx['best solution'][2:]]],
                        columns=['configuration', 'weight', 'value', 'squality', 'genotype']))

        if t == 'pso':
            print('____ starting PSO ____')
            for filename in tqdm(os.listdir('data/json files/json_configuration_pso_default/')):
                f = os.path.splitext(filename)[0]
                xx = run_configuration(f, knapsack, t, iters=pso_iters)
                all_configs = all_configs.append(
                    pd.DataFrame(
                        data=[[filename, *xx['best solution'][2:]]],
                        columns=['configuration', 'weight', 'value', 'squality', 'genotype']))

        if t == 'sa':
            print('____ starting SA ____')
            for filename in tqdm(os.listdir('data/json files/json_configuration_sa_default/')):
                f = os.path.splitext(filename)[0]
                xx = run_configuration(f, knapsack, t, iters=ga_iters)
                all_configs = all_configs.append(
                    pd.DataFrame(
                        data=[[filename, *xx['best solution'][2:]]],
                        columns=['configuration', 'weight', 'value', 'squality', 'genotype']))



class run_command_line(object):

    def __init__(self):
        error_message = """
        - - - - - - - - - - - - - - - - Run Evolutionary Computation - - - - - - - - - - - - - - - -

        Commands:                                arguments
            -configuration                         [name].json              file path
            -search_best_configuration             [ga | sa | pso]          type of computation
            
        Dependencies:
            numpy 
            pandas
            sys
            json
        
        """
        search_best_error_message = '''
    
        Argument passed to -search_best_configuration should be one of:
                    
            ga:  genetic algorithm
            sa:  simulated annealing
            pso: particle swarm intelligence
        '''
        config_error_message = '''
        
        Argument passed to -configuration should be a file path:
                    
            [name].json
        '''

        knapsack = pd.read_csv('data/knapsack_instance.csv', sep=';')

        if len(sys.argv) > 1:
            arg1 = sys.argv[1]

            if arg1 == '-configuration':
                if len(sys.argv) > 2:
                    arg2 = sys.argv[2]
                    # check file path

                    if 'ga' in arg2:
                        res = run_configuration(arg2, knapsack, 'ga')
                    elif 'sa' in arg2:
                        res = run_configuration(arg2, knapsack, 'sa')
                    elif 'pso' in arg2:
                        res = run_configuration(arg2, knapsack, 'pso')
                    else:
                        print(config_error_message)
                else:
                    print(config_error_message)




            elif arg1 == '-search_best_configuration':

                if len(sys.argv) > 2:
                    arg2 = sys.argv[2]
                    if arg2 == 'ga' or arg2 == 'sa' or arg2 == 'pso':
                        res = generate_all_reports([arg2])
                        write_best_config(arg2)
                    else:
                        print(search_best_error_message)
                else:
                    print(search_best_error_message)




            elif arg1 == '-h' or sys.argv[1] == '-help':
                print(error_message)

            else:
                print(error_message)

        else:
            print(error_message)



if __name__ == '__main__':
    run_command_line()






# Knapsack Problem

# In the 01 Knapsack problem, we are given a knapsack of fixed capacity C.
# We are also given a list of N objects, each having a
# weight W(I) and profit P(I). We can put any subset of the objects into
# the knapsack, as long as the total weight of our selection does not exceed C.
# We desire to maximize our total profit, which is the sum of the profits of
# each object we put into the knapsack.






# now = datetime.now()
# now.strftime("%Y-%m-%d %H:%M:%S")


# file_name = 'ga_default_01'
# with open('data/json files/json_configuration_ga_default/' + file_name + '.json') as file:
#     specs = json.load(file)
#
# file_name = 'pso_default_01'
# with open('data/json files/json_configuration_pso_default/' + file_name + '.json') as file:
#     specs = json.load(file)
#
# file_name = 'sa_default_01'
# with open('data/json files/json_configuration_sa_default/' + file_name + '.json') as file:
#     specs = json.load(file)
#
#
# ga = genetic_algorithm.compute_genetic_algorithm(knapsack, specs, iterations=10)
# fit = ga
# pso = particle_swarm_intelligence.PSO(knapsack, specs, iterations=10)
# fit = pso.fit
# sa = simulated_annealing.compute_sa_algorithm(knapsack, specs, iterations=10)
# fit = sa.fitness
#



# -------------------------------------------------------------- TEST RUN --------------------------------------------------------------x
# -------------------------------------------------------------- TEST RUN --------------------------------------------------------------x
# Run Genetic Algorithm
# generate_all_reports(['ga'])


# Simulated Annealing
# SA = generate_all_reports(['sa'])

# PSO
# pso = generate_all_reports(['pso'], pso_iters=1000)
# -------------------------------------------------------------- TEST RUN --------------------------------------------------------------x
# -------------------------------------------------------------- TEST RUN --------------------------------------------------------------x




#
#
#
#
# def run_search_best_configuration(knapsack, subclass, iters=10000):
#     start = timeit.default_timer()
#     file_names = []
#     all_configs = pd.DataFrame(columns=['configuration', 'fitness', 'weight', 'value', 'squality', 'genotype'])
#
#     if subclass == 'ga':
#         sc = 'Genetic Algorithm'
#         for filename in os.listdir('data/json files/json_configuration_ga_default/'):
#             file_names.append(filename)
#             with open('data/json files/json_configuration_ga_default/' + filename) as file:
#                 specs = json.load(file)
#             ga = genetic_algorithm.compute_genetic_algorithm(knapsack, specs, iterations=iters)
#             fit = ga
#             cf = fit.iloc[pd.Index(fit.fitness).get_loc(max(fit.fitness)),]
#             all_configs = all_configs.append(
#                 pd.DataFrame([['ga_default__'] + [cf['fitness']] + [cf['weight']] + [cf['value']] + [cf['squality']] + [
#                     cf['genotype']]],
#                              columns=['configuration', 'fitness', 'weight', 'value', 'squality', 'genotype']))
#
#     elif subclass == 'pso':
#         sc = 'Particle Swarm Intelligence'
#         for filename in os.listdir('data/json files/json_configuration_pso_default/'):
#             file_names.append(filename)
#             with open('data/json files/json_configuration_pso_default/' + filename) as file:
#                 specs = json.load(file)
#             pso = particle_swarm_intelligence.PSO(knapsack, specs, iterations=iters)
#             fit = pso.fit
#             cf = fit.iloc[pd.Index(fit.fitness).get_loc(max(fit.fitness)),]
#             all_configs = all_configs.append(
#                 pd.DataFrame([['ga_default__'] + [cf['fitness']] + [cf['weight']] + [cf['value']] + [cf['squality']] + [
#                     cf['genotype']]],
#                              columns=['configuration', 'fitness', 'weight', 'value', 'squality', 'genotype']))
#
#     elif subclass == 'sa':
#         sc = 'Simulated Annealing'
#         for filename in os.listdir('data/json files/json_configuration_sa_default/'):
#             file_names.append(filename)
#             with open('data/json files/json_configuration_sa_default/' + filename) as file:
#                 specs = json.load(file)
#             sa = simulated_annealing.compute_sa_algorithm(knapsack, specs, iterations=iters)
#             fit = sa.fitness
#             cf = fit.iloc[pd.Index(fit.fitness).get_loc(max(fit.fitness)),]
#             all_configs = all_configs.append(
#                 pd.DataFrame([['ga_default__'] + [cf['fitness']] + [cf['weight']] + [cf['value']] + [cf['squality']] + [
#                     cf['genotype']]],
#                              columns=['configuration', 'fitness', 'weight', 'value', 'squality', 'genotype']))
#
#     stop = timeit.default_timer()
#     results = {
#         'subclass': sc,
#         'all_configs': all_configs,
#         'miniseconds': round((stop - start) * 1000, 4)
#     }
#     return results
#
#
#
