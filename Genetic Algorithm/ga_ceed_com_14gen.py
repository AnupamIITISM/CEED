# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 23:41:34 2019

@author: drshr
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 10:38:32 2019

@author: drshr
"""

import pandas as pd
import numpy as np
import ga_ceed_14gen
import matplotlib.pyplot as plt



pdemand = 2000
prob_mutation = 0.2
no_of_itr = 100
best_res = np.zeros([no_of_itr,2])
#gen_no = 6
population_size = 8

data_14_gen_df = pd.read_excel('ceed_data_14_gen.xlsx')
gen_no, parameter_size = data_14_gen_df.shape
half_popul = population_size//2

#print(data_14_gen_df)

# Initial population
popln = ga_ceed_14gen.initiate_population_random(data_14_gen_df, population_size, gen_no, pdemand)
print(popln)
for i in range(no_of_itr):
    ####    Cost calculation of population
    popln_cost = ga_ceed_14gen.calculate_population_cost(popln, data_14_gen_df, pdemand)
    #    print(popln_cost)
    
    ##  Parent Selection
    parents = ga_ceed_14gen.parent_selection(popln_cost)
    #    print(parents)
    
    ####    Crossover
    cross_childs = ga_ceed_14gen.whole_linear_crossover(parents, data_14_gen_df, pdemand)
    #    print(cross_childs)
    
    ####    Mutation
    mutate_childs = ga_ceed_14gen.mutation(cross_childs, data_14_gen_df, pdemand, prob_mutation)
    #    print(mutate_childs)
    
    #### Combination of Childs and Parents to generate new population
    child_parents = np.zeros([population_size, gen_no+1])
    
    child_parents[:half_popul] = parents
    child_parents[half_popul:] = mutate_childs
    
    popln = np.copy(child_parents[:,:-1])
    
    best_solution = ga_ceed_14gen.best_solution(child_parents)
    
    best_res[i][0] = i
    best_res[i][1] = best_solution[-1]

print(best_solution)
best_cost = best_solution[-1]
init_cost = best_res[0][1]


    
plt.plot(best_res)
plt.ylim(best_cost, init_cost)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()    
    
    




    

