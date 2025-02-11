# code to make the possible 2*2*4 multielement alloy structure and save the feature of the alloy into csv file
# The featurization process is described in paper

import numpy as np
import copy
import random
import sys
from itertools import product
from collections import defaultdict
from sub_codes import multiplicity
import csv
import fileinput

i=0
mults=np.zeros(390625)
feature=np.zeros((390625,15))
possible_surface=list(product(['Pt','Ru','Cu','Ni','Fe'],repeat=8))

for comb in possible_surface:
    first={}
    second= {}
    third= {}
    for sp in ['Pt','Ru','Cu','Ni','Fe']:
        first[sp] = 0
        second[sp] = 0
        third[sp] = 0
    third[comb[0]]+=1
    third[comb[2]]+=1
    third[comb[3]]+=1
    first[comb[4]]+=1
    second[comb[5]]+=2
    second[comb[6]]+=2
    second[comb[7]]+=2

    firstvalues=list(first.values())
    secondvalues=list(second.values())
    thirdvalues=list(third.values())

    firstval=[x for x in firstvalues if x!=0]
    secondval=[x for x in secondvalues if x!=0]
    thirdval=[x for x in thirdvalues if x!=0]

    firstmult=multiplicity(1,firstval)
    secondmult=multiplicity(6, secondval)
    thirdmult=multiplicity(3,thirdval)


    totalmult=firstmult*secondmult*thirdmult
    feature[i]=np.array(firstvalues+secondvalues+thirdvalues)
    mults[i]=totalmult

    i+=1

file=open('index_metal_top.csv','w',newline='')
with file:
    write=csv.writer(file)
    write.writerows(possible_surface)
output=np.c_[feature,mults]
np.savetxt('possibletop.csv',output,fmt=['%d']*16, delimiter=',')
