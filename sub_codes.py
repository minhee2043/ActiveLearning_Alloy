import numpy as np
from math import factorial
import itertools as it

def count_metals(metals, nMetals):
    '''return list of counts of each  metal in input relative to instance 'metals'
    metals   list of ints
    nMetals  number of metals in alloy'''
    counts = [0]*nMetals
    for metal in metals:
        counts[metal] += 1
    return counts

def multiplicity(nAtoms, nEachMetal):
    '''nAtoms       int                 number of atoms in zone
    nEachMetal   list of ints        number of each metal in zone'''
    product = 1
    for nMetal in nEachMetal:
        product *= factorial(nMetal)
    return factorial(nAtoms)/product
