# code to make the possible 2*2*4 multielement alloy structure and save the feature of the alloy into csv file
# The featurization process is described in paper

from ase import Atoms
from ase.visualize import view
from ase.build import bulk, molecule, fcc111
from ase.calculators.vasp import Vasp
from matplotlib import pyplot as plt
from ase.optimize import BFGS
from ase.constraints import FixAtoms
import numpy as np
import copy
from itertools import combinations
from ase.optimize import BFGS
from ase.io import Trajectory, read, write
import ase.db
from ase.db import connect
from ase.build import add_adsorbate
import random
import sys
from itertools import product
from collections import defaultdict
from helperMethods import multiplicity
import csv
import fileinput

i=0
mults=np.zeros(6561)
feature=np.zeros((6561,21))
possible_surface=list(product(['Ni','Co','Fe'],repeat=8))

for comb in possible_surface:
    first={}
    second= {}
    third= {}
    fourth= {}
    fifth= {}
    sixth= {}
    seventh = {}
    for sp in ['Ni','Co','Fe']:
        first[sp] = 0
        second[sp] = 0
        third[sp] = 0
        fourth[sp] = 0
        fifth[sp] = 0
        sixth[sp] = 0
        seventh[sp] = 0

    first[comb[4]]+=1
    first[comb[5]]+=1
    second[comb[7]]+=4
    third[comb[0]]+=1
    third[comb[1]]+=1
    fourth[comb[6]]+=2
    fifth[comb[2]]+=1
    sixth[comb[4]]+=1
    sixth[comb[5]]+=1
    seventh[comb[3]]+=2
    
    firstvalues=list(first.values())
    secondvalues=list(second.values())
    thirdvalues=list(third.values())
    fourthvalues=list(fourth.values())
    fifthvalues=list(fifth.values())
    sixthvalues=list(sixth.values())
    seventhvalues=list(seventh.values())
    
    firstval=[x for x in firstvalues if x!=0]
    secondval=[x for x in secondvalues if x!=0]
    thirdval=[x for x in thirdvalues if x!=0]
    fourthval=[x for x in fourthvalues if x!=0]
    fifthval=[x for x in fifthvalues if x!=0]
    sixthval=[x for x in sixthvalues if x!=0]
    seventhval=[x for x in seventhvalues if x!=0]
    
    firstmult=multiplicity(2,firstval)
    secondmult=multiplicity(4, secondval)
    thirdmult=multiplicity(2,thirdval)
    fourthmult=multiplicity(2, fourthval)
    fifthmult=multiplicity(1, fifthval)
    sixthmult=multiplicity(2, sixthval)
    seventhmult=multiplicity(2, seventhval)
    

    totalmult=firstmult*secondmult*thirdmult*fourthmult*fifthmult*sixthmult*seventhmult
    feature[i]=np.array(firstvalues+secondvalues+thirdvalues+fourthvalues+fifthvalues+sixthvalues+seventhvalues)
    mults[i]=totalmult

    i+=1

file=open('index_metal.csv','w',newline='')
with file:
    write=csv.writer(file)
    write.writerows(possible_surface)
output=np.c_[feature,mults]
np.savetxt('possibleFp.csv',output,fmt=['%d']*22, delimiter=',')

