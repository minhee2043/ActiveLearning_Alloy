# activity calculation according to adsorption energy
from statistics import mean
from itertools import repeat
from csv import writer
from csv import reader
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

kb=8.617333262*math.pow(10,-5)
T=300
E_opt=-0.27

possible_contributions = [i/100 for i in range(0,101,5)]
all_model_combs = it.product(possible_contributions, repeat=5)
possibleratio = [comb for comb in all_model_combs if math.isclose(sum(comb),1, rel_tol=1e-06)]

possibleratio= possibleratio[10500:]
f=open('act_10500_10626.txt','w')
"""
rows=[]
for i in range(0,35):
    file=open(f'batch5/GPR_group{i}.csv')
    csvreader=csv.reader(file)
    for row in csvreader:
        rows.append(row)
"""

def diff(x):
    return math.exp(-abs(x+0.27)/(kb*T))

print('start')
def func(data, ratio):
    #print(ratio)
    #print(row)
    data['Pt']=data.iloc[:,[0,5,10,15,20]].sum(axis=1)
    data['Ru']=data.iloc[:,[1,6,11,16,21]].sum(axis=1)
    data['Cu']=data.iloc[:,[2,7,12,17,22]].sum(axis=1)
    data['Ni']=data.iloc[:,[3,8,13,18,23]].sum(axis=1)
    data['Fe']=data.iloc[:,[4,9,14,19,24]].sum(axis=1)
    data['Ptnew']=data['Pt'].apply(lambda x:math.pow(ratio[0],x))
    data['Runew']=data['Ru'].apply(lambda x:math.pow(ratio[1],x))
    data['Cunew']=data['Cu'].apply(lambda x:math.pow(ratio[2],x))
    data['Ninew']=data['Ni'].apply(lambda x:math.pow(ratio[3],x))
    data['Fenew']=data['Fe'].apply(lambda x:math.pow(ratio[4],x))
    data['possibility']=data.Ptnew*data.Runew*data.Cunew*data.Ninew*data.Fenew
    final= data.possibility*data.iloc[:,26].apply(diff)*data.iloc[:,25]
    #print(final)
    return final

def parallel_df(df, func, n_cores,ratio):
    df_split=np.array_split(df,n_cores)
    pool=Pool(n_cores)
    df=pd.concat(pool.starmap(func,zip(df_split,repeat(ratio))))
    pool.close()
    pool.join()
    return df

activities=[]
for ratio in possibleratio:
    activity=0
    for i in range(0,35):
        #print('i: ', i)
        df=pd.read_csv(f'../stable_batch20/GPR_group{i}.csv',header=None)
        df=parallel_df(df,func,n_cores=24,ratio=ratio)
        activity += df.sum()
        #print(activity)
    print(ratio)
    f.write(str(ratio))
    print(activity)
    print('\n')
    f.write(str(activity))
    f.write('\n')
    activities.append(activity)


#print(activities)
#print(len(activities))
max_value=max(activities)
print('\n')
print('\n')
print(max_value)
f.write(str(max_value))
max_ind=activities.index(max_value)
print(possibleratio[max_ind])
f.write(str(possibleratio[max_ind]))
f.close()
