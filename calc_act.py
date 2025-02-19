# activity calculation from GPR-prediced adsorption energies
from statistics import mean                   
from itertools import repeat                  
from csv import writer, reader                
import math                                   
import sys                                    
import itertools as it                        
import numpy as np                           
from math import factorial                  
from collections import Counter               
import pandas as pd                         
from more_itertools import distinct_permutations  
import matplotlib.pyplot as plt               
import csv                                   
from scipy.optimize import minimize           
from multiprocesspandas import applyparallel  
from multiprocessing import Pool              

# Define constants for calculations
kb = 8.617333262 * math.pow(10, -5)  # Boltzmann constant 
T = 300                              
E_opt = -0.27                        # Optimal adsorption energy in eV

# Generate possible contribution ratios (in 5% increments) that sum to 1
possible_contributions = [i/100 for i in range(0, 101, 5)]  # From 0 to 1.0 in steps of 0.05
all_model_combs = it.product(possible_contributions, repeat=5)  # Generate all combinations for 5 metals
possibleratio = [comb for comb in all_model_combs if math.isclose(sum(comb), 1, rel_tol=1e-06)]  

# Open file to write results
f = open('activity.txt', 'w')

# Define function to calculate activity based on adsorption energy difference from optimal
def diff(x):
    """
    Calculate activity factor based on how close the adsorption energy is to optimal value (-0.27 eV)
    Uses an exponential decay based on the difference
    """
    return math.exp(-abs(x+0.27)/(kb*T))

print('start')

# Main function to calculate activity for a given ratio of metal contributions
def func(data, ratio):
    """
    Calculate the activity for a given dataset and metal contribution ratios
    
    Parameters:
    - data: DataFrame containing metal configurations and energies
    - ratio: Tuple of 5 values representing contribution ratios of Pt, Ru, Cu, Ni, Fe
    
    Returns:
    - Final activity values for each configuration
    """
    # Sum the contributions of each metal type across different positions
    data['Pt'] = data.iloc[:, [0, 5, 10, 15, 20]].sum(axis=1)
    data['Ru'] = data.iloc[:, [1, 6, 11, 16, 21]].sum(axis=1)
    data['Cu'] = data.iloc[:, [2, 7, 12, 17, 22]].sum(axis=1)
    data['Ni'] = data.iloc[:, [3, 8, 13, 18, 23]].sum(axis=1)
    data['Fe'] = data.iloc[:, [4, 9, 14, 19, 24]].sum(axis=1)
    
    # Calculate probability factors based on metal counts and ratios
    data['Ptnew'] = data['Pt'].apply(lambda x: math.pow(ratio[0], x))
    data['Runew'] = data['Ru'].apply(lambda x: math.pow(ratio[1], x))
    data['Cunew'] = data['Cu'].apply(lambda x: math.pow(ratio[2], x))
    data['Ninew'] = data['Ni'].apply(lambda x: math.pow(ratio[3], x))
    data['Fenew'] = data['Fe'].apply(lambda x: math.pow(ratio[4], x))
    
    # Calculate overall possibility (probability) of each configuration
    data['possibility'] = data.Ptnew * data.Runew * data.Cunew * data.Ninew * data.Fenew
    
    # Calculate final activity by multiplying possibility by activity factor and weight
    # Column 26 contains adsorption energy, Column 25 contains configuration weight
    final = data.possibility * data.iloc[:, 26].apply(diff) * data.iloc[:, 25]
    
    return final

# Function to parallelize dataframe processing across multiple cores
def parallel_df(df, func, n_cores, ratio):
    """
    Split dataframe and process in parallel
    
    Parameters:
    - df: DataFrame to process
    - func: Function to apply
    - n_cores: Number of CPU cores to use
    - ratio: Metal contribution ratios to pass to func
    
    Returns:
    - Processed DataFrame
    """
    df_split = np.array_split(df, n_cores)  # Split dataframe into chunks
    pool = Pool(n_cores)                    # Create process pool
    df = pd.concat(pool.starmap(func, zip(df_split, repeat(ratio))))  # Process in parallel
    pool.close()
    pool.join()
    return df

# Calculate activity for each possible ratio combination
activities = []
for ratio in possibleratio:
    activity = 0
    # Process each group of data (35 groups total)
    for i in range(0, 35):
        df = pd.read_csv(f'../GPR_batch20/GPR_group{i}.csv', header=None)
        df = parallel_df(df, func, n_cores=24, ratio=ratio)
        activity += df.sum()  # Sum all activities for this group
    
    # Print and save results for current ratio
    print(ratio)
    f.write(str(ratio))
    print(activity)
    print('\n')
    f.write(str(activity))
    f.write('\n')
    activities.append(activity)

# Find and output the maximum activity and corresponding ratio
max_value = max(activities)
print('\n\n')
print(max_value)
f.write(str(max_value))
max_ind = activities.index(max_value)
print(possibleratio[max_ind])  # Print the optimal ratio
f.write(str(possibleratio[max_ind]))
f.close()
