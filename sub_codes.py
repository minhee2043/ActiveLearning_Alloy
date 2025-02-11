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

def count_atoms(userSymbols,refSymbols):
    '''return list of the number each symbol in userSymbols occurs in the order given by
    refSymbols and thus only compares symbols in refSymbols'''
    nRef = len(refSymbols)
    counts = [0]*nRef

    if isinstance(userSymbols,list): # if userSymbols is a list
        if not userSymbols: # if list is empty return an empty list
            return []
        for symbol in userSymbols:
            for i,refSymbol in enumerate(refSymbols):
                if symbol == refSymbol:
                    counts[i] += 1
    else: # if userSymbols is a string
        for i,refSymbol in enumerate(refSymbols):
            if userSymbols == refSymbol:
                counts[i] += 1
    return counts

def sortMetals(metalList, metals):
    '''return list of 'metalList' sorted by the order given in 'metals' '''
    nMetals = len(metals) # 5
    n = len(metalList) # e.g. 3
    metalNum = [0]*n
    for i in range(n): # 3
        for j in range(nMetals): # 5
            if metalList[i] == metals[j]:
                metalNum[i] = j # assign a number to each metal symbol
    metalNum = sorted(metalNum) # sort the numbers
    sortedMetals = ['']*n
    for i in range(n): # 3
        for j in range(nMetals): # 5
            if metalNum[i] == j:
                sortedMetals[i] = metals[j] # convert back to symbols
    return sortedMetals

def remove_zero_columns(X):
    '''return X without zero columns and the ids of the removed columns'''
    keepIds, removeIds = [], []
    for i,column in enumerate(X.T):
        if np.array_equal(column,np.zeros(len(X))):
            removeIds.append(i) # index for columns that are pure zeros
        else:
            keepIds.append(i) # index for columns that are not pure zeros
    return X[:,keepIds], removeIds, keepIds

def unique(nMetals,nSites):
    return (factorial(nMetals+nSites-1)/(factorial(nSites)*factorial(nMetals-1)))

def multiplicity(nAtoms, nEachMetal):
    '''nAtoms       int                 number of atoms in zone
    nEachMetal   list of ints        number of each metal in zone'''
    product = 1
    for nMetal in nEachMetal:
        product *= factorial(nMetal)
    return factorial(nAtoms)/product
