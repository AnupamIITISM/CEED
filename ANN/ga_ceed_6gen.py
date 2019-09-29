# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 06:36:18 2019

@author: drshr

***** Approach -------
1. Generate initial Candidates -- there may be 2 approaches
    A. For 6 no of generators, randomly generate 5 data within their range, 
    then get the last value calculating from the total load
    B. Initialize all the values to the same values
    
    -- We can combine A and B to generate the initial population
    
2. Apply fitness function(minimize cost) and 

3. select the candidates (Parents) for the mating pool in probabilistic manner

4. Do crossover

5. Do mutation

6. Repeat  Steps 2 to 5 for a number of iterations (say 1000)

3. 
"""
import pandas as pd
import numpy as np
import random

# calculation of cost for individual candidate
def calc_candidate_fuel_cost(candidate, data_6_gen_df):
    # calculate the fuel cost
    fuel_cost = 0
    a_params = data_6_gen_df[data_6_gen_df.columns[0:3]]
    for i in range(candidate.size):
        pi = candidate[i]
        generator_cost = a_params.values[i][0]*pi*pi  + a_params.values[i][1]*pi + a_params.values[i][2]
        fuel_cost = fuel_cost + generator_cost
    return fuel_cost

def calc_candidate_emission_cost(candidate, data_6_gen_df):        
    # calculate the fuel emission
    emission_cost = 0
    alpha_params = data_6_gen_df[data_6_gen_df.columns[3:6]]
    for i in range(candidate.size):
        pi = candidate[i]
        generator_emi_cost = alpha_params.values[i][0]*pi*pi  + alpha_params.values[i][1]*pi + alpha_params.values[i][2]
        emission_cost = emission_cost + generator_emi_cost
    return emission_cost

def calc_candidate_total_cost(candidate, data_6_gen_df, pdemand):
    fuel_cost = calc_candidate_fuel_cost(candidate, data_6_gen_df)
    emission_cost = calc_candidate_emission_cost(candidate, data_6_gen_df)
    hi_val = calc_hi(data_6_gen_df, candidate, pdemand)
    total_cost = fuel_cost + hi_val*emission_cost
    return total_cost

def calc_hi(data_6_gen_df, candidate, pdemand):
    a_params = data_6_gen_df[data_6_gen_df.columns[0:3]]
    alpha_params = data_6_gen_df[data_6_gen_df.columns[3:6]]
    pi_params = data_6_gen_df[data_6_gen_df.columns[6:8]]
    h = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
    for i in range(6):
        # Step 1
        pimax = pi_params.values[i][1]
        eipimax = a_params.values[i][0]*pimax*pimax  + a_params.values[i][1]*pimax + a_params.values[i][2]
        fipimax = alpha_params.values[i][0]*pimax*pimax  + alpha_params.values[i][1]*pimax + alpha_params.values[i][2]
        h[i] = fipimax/eipimax
    # Step 2 and 3
    h_copy = np.copy(h)
    ptot = 0.0
    idx = 0
    while ptot < pdemand :
        idx = np.argmin(h_copy)
        ptot += pi_params.values[idx][1]
        h_copy[idx] = 99999 # Set to a very high value
    #Step 4
    penalty_factor = h[idx]    
    return penalty_factor

# -----------------
    
def initiate_population_random(data_6_gen_df, pop_size, gen_no, pdemand):
    pi_params = data_6_gen_df[data_6_gen_df.columns[6:8]]
    initial_pop = np.zeros([pop_size, gen_no])
    cand = np.zeros(gen_no)
    for i in range(pop_size):
        cand_created = False
        while not cand_created:            
            tot_load = 0.0
            for j in range(gen_no-1):
                pimin = pi_params.values[j][0]
                pimax = pi_params.values[j][1]
                cand[j] = random.randrange(pimin,pimax, step=1)
                tot_load += cand[j]
                if tot_load > pdemand:
                    tot_load = 0.0
                    cand_created = False
                    break
            last_gen_load = pdemand - tot_load
            if last_gen_load > pi_params.values[gen_no-1][0] and last_gen_load < pi_params.values[gen_no-1][1]:
                cand[gen_no-1] = last_gen_load
                cand_created = True
            else:
                cand_created = False
        
        initial_pop[i] = cand[:]
    return initial_pop
            
def calculate_population_cost(popln, data_6_gen_df, pdemand):
    pop_size, gen_no = popln.shape
    initial_pop_cost = np.zeros([pop_size, gen_no+1])
    for i in range(pop_size):
        for j in range(gen_no):
            initial_pop_cost[i][j] = popln[i][j]
        initial_pop_cost[i][gen_no] = calc_candidate_total_cost(popln[i], data_6_gen_df, pdemand)
    return initial_pop_cost    
        
def sortCost(val): 
    return val[-1]  

def parent_selection(popln_cost):
    popln_cost_list = popln_cost.tolist()
    popln_cost_list.sort(key = sortCost)
    selected_parents = popln_cost_list[:4]
    selected_parents_arr = np.array(selected_parents)
    return selected_parents_arr

def best_solution(popln_cost):
    popln_cost_list = popln_cost.tolist()
    popln_cost_list.sort(key = sortCost)
    minimum = popln_cost_list[0]
    return minimum

def whole_linear_crossover(parent_popln_cost, data_6_gen_df, pdemand):
    child_popln_cost = np.copy(parent_popln_cost)
    child_popln_cost_copy = np.copy(parent_popln_cost)
    # Randomly choose 2 parents
    i = random.randrange(0, 4, step=1)
    j = random.randrange(0, 4, step=1)
    while i == j:
        j = random.randrange(0, 4, step=1)
    child_popln_cost_copy[0] = 0.5 * parent_popln_cost[i] + 0.5 * parent_popln_cost[j] 
    child_popln_cost_copy[1] = 1.5 * parent_popln_cost[i] - 0.5 * parent_popln_cost[j] 
    child_popln_cost_copy[2] = -0.5 * parent_popln_cost[i] + 1.5 * parent_popln_cost[j]    
    child_popln_cost_copy[3] = 0.75 * parent_popln_cost[i] + .25 * parent_popln_cost[j] 
    
    child_popln_cost_copy = calculate_population_cost(child_popln_cost_copy[:,:-1],data_6_gen_df, pdemand)
    child_popln_cost_copy_list = child_popln_cost_copy.tolist()
    child_popln_cost_copy_list.sort(key = sortCost)
    selected_child_pop = child_popln_cost_copy_list[:2]
    selected_child_pop_arr = np.array(selected_child_pop)
    
    k = random.randrange(0, 4, step=1)
    while k == i or k==j:
        k = random.randrange(0, 4, step=1)
    l = random.randrange(0, 4, step=1)
    while l == i or l == j or l==k:
        l = random.randrange(0, 4, step=1)
        
    child_popln_cost_copy[0] = 0.5 * parent_popln_cost[k] + 0.5 * parent_popln_cost[l] 
    child_popln_cost_copy[1] = 1.5 * parent_popln_cost[k] - 0.5 * parent_popln_cost[l] 
    child_popln_cost_copy[2] = -0.5 * parent_popln_cost[k] + 1.5 * parent_popln_cost[l]    
    child_popln_cost_copy[3] = 0.75 * parent_popln_cost[k] + 0.25 * parent_popln_cost[l] 
        
    child_popln_cost_copy = calculate_population_cost(child_popln_cost_copy[:,:-1],data_6_gen_df, pdemand)
    child_popln_cost_copy_list = child_popln_cost_copy.tolist()
    child_popln_cost_copy_list.sort(key = sortCost)
    selected_child_pop_l = child_popln_cost_copy_list[:2]
    selected_child_pop_arr_1 = np.array(selected_child_pop_l)
    
    child_popln_cost[:2] = selected_child_pop_arr
    child_popln_cost[2:] = selected_child_pop_arr_1
    
    return child_popln_cost
    

def mutation(child_popln_cost, data_6_gen_df, pdemand, prob_mutation):    
    pi_params = data_6_gen_df[data_6_gen_df.columns[6:8]]
    pop_size, gen_no = child_popln_cost.shape
    gen_no = gen_no - 1
    mutated_popln_cost = np.copy(child_popln_cost)
    for pop in range(pop_size):
        chance = random.randrange(0, 100, step=1)
        #    Mutation can occour with the mutation probability 
        if(chance < prob_mutation * 100):
            cand = mutated_popln_cost[pop]
            # Randomly select 2 generator
            i = random.randrange(0, gen_no, step=1)
            j = random.randrange(0, gen_no, step=1)
            while i == j:
                j = random.randrange(0, 4, step=1)
            pi = cand[i]
            pj = cand[j]
            pimax = pi_params.values[i][1]
            pjmin = pi_params.values[j][0]
            delta_incr = min((pimax-pi),(pj-pjmin))*prob_mutation
            pi = pi + delta_incr
            pj = pj - delta_incr
            cand[i] = pi
            cand[j] = pj
            mutated_popln_cost[pop] = cand
            
    mutated_popln_cost = calculate_population_cost(mutated_popln_cost[:,:-1],data_6_gen_df, pdemand)
        
    return mutated_popln_cost
   
    
    
    
    
    