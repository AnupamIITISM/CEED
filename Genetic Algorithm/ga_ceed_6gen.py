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

def calc_candidate_total_cost(candidate, data_6_gen_df, h_val):
    fuel_cost = calc_candidate_fuel_cost(candidate, data_6_gen_df)
    emission_cost = calc_candidate_emission_cost(candidate, data_6_gen_df)
    total_cost = fuel_cost + h_val*emission_cost
    return total_cost

