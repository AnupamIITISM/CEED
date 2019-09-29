# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 14:00:07 2019

@author: drshr
"""

# Collect data from 6 Generator algorithm

import pandas as pd
import numpy as np
import ga_ceed_6gen
import random
import matplotlib.pyplot as plt

result_list = np.zeros([45, 8])

j = 0

for pdemand in range(400, 1300, 20):
    print(pdemand)
    prob_mutation = 0.2
    no_of_itr = 500
    best_res = np.zeros([no_of_itr,2])
    #gen_no = 6
    population_size = 8
    
    data_6_gen_df = pd.read_excel('ceed_data_6_gen.xlsx')
    gen_no, parameter_size = data_6_gen_df.shape
    half_param = parameter_size//2
    
    #print(data_6_gen_df)
    
    # Initial population
    popln = ga_ceed_6gen.initiate_population_random(data_6_gen_df, population_size, gen_no, pdemand)
    #print(popln)
    for i in range(no_of_itr):
        print(i)
        ####    Cost calculation of population
        popln_cost = ga_ceed_6gen.calculate_population_cost(popln, data_6_gen_df, pdemand)
        #    print(popln_cost)
        
        ##  Parent Selection
        parents = ga_ceed_6gen.parent_selection(popln_cost)
        #    print(parents)
        
        ####    Crossover
        cross_childs = ga_ceed_6gen.whole_linear_crossover(parents, data_6_gen_df, pdemand)
        #    print(cross_childs)
        
        ####    Mutation
        mutate_childs = ga_ceed_6gen.mutation(cross_childs, data_6_gen_df, pdemand, prob_mutation)
        #    print(mutate_childs)
        
        #### Combination of Childs and Parents to generate new population
        child_parents = np.zeros([population_size, gen_no+1])
        
        child_parents[:half_param] = parents
        child_parents[half_param:] = mutate_childs
        
        popln = np.copy(child_parents[:,:-1])
        
        best_solution = ga_ceed_6gen.best_solution(child_parents)
        
        best_res[i][0] = i
        best_res[i][1] = best_solution[-1]
    
    print(best_solution)
    result_list[j][0] = pdemand
    result_list[j][1:] = best_solution
    j = j+1
    
print(result_list)
best_solution_df = pd.DataFrame(result_list)
best_solution_df.to_excel('ga_data_4_ann.xlsx')