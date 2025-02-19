# code to make the possible 2*2*4 multielement alloy structure and save the feature of the alloy into csv file
# The featurization process is described in paper

# Import necessary libraries
from ase import Atoms                       # For atomic structure creation
from ase.visualize import view              # For visualizing atomic structures
from ase.build import bulk, molecule, fcc111 # For building crystal structures
from ase.calculators.vasp import Vasp       # VASP calculator interface
from matplotlib import pyplot as plt        # For plotting
from ase.optimize import BFGS               # For geometry optimization
from ase.constraints import FixAtoms        # For constraining atoms
import numpy as np                          # For numerical operations
import copy                                 # For deep copying objects
from itertools import combinations, product  # For generating combinations and products
from ase.io import Trajectory, read, write  # For reading/writing atomic structures
import ase.db                               # ASE database
from ase.db import connect                  # For connecting to ASE database
from ase.build import add_adsorbate         # For adding adsorbates
import random                               # For random number generation
import sys                                  # For system-specific parameters
from collections import defaultdict         # For defaultdict data structure
from helperMethods import multiplicity      # Custom function for calculating multiplicity
import csv                                  # For CSV operations
import fileinput                            # For file input operations

# Initialize counters and arrays
i = 0
mults = np.zeros(390625)                     # Array to store multiplicity values (5^8 = 390625 possible structures)
feature = np.zeros((390625, 26))             # Array to store features (26 features per structure)

# Generate all possible combinations of Pt,Ru,Cu,Ni,Fe atoms in 8 positions
possible_surface = list(product(['Pt', 'Ru', 'Cu', 'Ni', 'Fe'], repeat=8))

# Iterate through all possible configurations
for comb in possible_surface:
    # Initialize dictionaries to count atom types in different positions/environments
    first = {}
    second = {}
    third = {}
    fourth = {}
    fifth = {}
    
    # Initialize counters for each element in each position
    for sp in ['Pt', 'Ru', 'Cu', 'Ni', 'Fe']:
        first[sp] = 0
        second[sp] = 0
        third[sp] = 0
        fourth[sp] = 0
        fifth[sp] = 0

    # Count atoms in specific positions according to the featurization scheme
    # This represents different coordination environments in the alloy structure
    fifth[comb[0]]+=1
    fifth[comb[1]]+=1
    fifth[comb[2]]+=1
    third[comb[2]]+=1
    third[comb[3]]+=2
    first[comb[4]]+=1
    second[comb[4]]+=2
    first[comb[5]]+=1
    second[comb[5]]+=2
    first[comb[6]]+=1
    fourth[comb[6]]+=1
    second[comb[7]]+=2
    fourth[comb[7]]+=2
    
    # Extract values (counts) from each dictionary
    firstvalues = list(first.values())
    secondvalues = list(second.values())
    thirdvalues = list(third.values())
    fourthvalues = list(fourth.values())
    fifthvalues = list(fifth.values())
    
    # Filter out zero counts
    firstval = [x for x in firstvalues if x != 0]
    secondval = [x for x in secondvalues if x != 0]
    thirdval = [x for x in thirdvalues if x != 0]
    fourthval = [x for x in fourthvalues if x != 0]
    fifthval = [x for x in fifthvalues if x != 0]

    # Calculate multiplicity for each position group
    # Multiplicity represents the number of equivalent configurations
    firstmult = multiplicity(2, firstval)
    secondmult = multiplicity(4, secondval)
    thirdmult = multiplicity(2, thirdval)
    fourthmult = multiplicity(2, fourthval)
    fifthmult = multiplicity(1, fifthval)

    
    # Calculate total multiplicity by multiplying individual multiplicities
    totalmult = firstmult * secondmult * thirdmult * fourthmult * fifthmult 
    
    # Store feature vector and multiplicity
    feature[i] = np.array(firstvalues + secondvalues + thirdvalues + fourthvalues + fifthvalues)
    mults[i] = totalmult
    i += 1

# Save the atom configurations to a CSV file
file = open('index_metal.csv', 'w', newline='')
with file:
    write = csv.writer(file)
    write.writerows(possible_surface)

# Combine features and multiplicities and save to CSV
output = np.c_[feature, mults]
np.savetxt('possibleFp.csv', output, fmt=['%d'] * 26, delimiter=',')

