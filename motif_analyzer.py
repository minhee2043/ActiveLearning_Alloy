from ase import atoms
from ase.build import fcc111
from ase.visualize import view
from itertools import combinations
import sys, copy
import numpy as np
from collections import defaultdict
from ase.build import add_adsorbate
import ase.db
import ase.neighborlist as ase_neighborlist


def get_adsorbates(slab):
    ads_atoms = []
    for atom in slab:
        if atom.tag == 0:
            ads_atoms.append(atom)
    return ads_atoms   

def get_motif(slab, substrate_species = ['Pt', 'Ru', 'Cu', 'Ni', 'Fe'], adsorbate_specie='H'):
    ads_atoms = get_adsorbates(slab)
    direct_ads_atom = [ads_ for ads_ in ads_atoms if ads_.symbol in adsorbate_specie]
    pos = direct_ads_atom[0].position
    surface_atoms = [surf.index for surf in slab if surf.tag ==1]
    subsurface_atoms = [surf.index for surf in slab if surf.tag ==2]
    nat_cutoff = ase_neighborlist.natural_cutoffs(slab)
    n1 = ase_neighborlist.NeighborList(nat_cutoff, self_interaction=False, bothways=True)
    n1.update(slab)
    indices, offsets = n1.get_neighbors(ads_atoms[0].index)
    indices = np.delete(indices, np.where((indices >16)))
    print(indices)
    
    first_neighbor = []
    first_neighbor_surf=defaultdict(int)
    second_neighbor_surf = defaultdict(int)
    second_neighbor_sub = defaultdict(int)
    fourth_neighbor_surf = defaultdict(int)
    fifth_neighbor_sub = defaultdict(int)
    sixth_neighbor_surf = defaultdict(int)

    distance = [slab.get_distance(16, i) for i in surface_atoms]
    mx = max(distance)
    new_dis = set(distance)
    new_dis.remove(max(new_dis))
    mx2 = max(new_dis)
    
    #initialize counts
    for sp in substrate_species:
        first_neighbor_surf[sp] = 0
        second_neighbor_surf[sp] = 0
        second_neighbor_sub[sp] = 0
        fourth_neighbor_surf[sp] = 0
        fifth_neighbor_sub[sp] = 0
        sixth_neighbor_surf[sp] = 0
    
    for i in indices:
        first_neighbor.append(i)
        first_neighbor_surf[str(slab[i].symbol)] +=1

    if len(first_neighbor) == 1:
        total_second_neighbor = set()
        for i in first_neighbor:
            indices_, offsets_ = n1.get_neighbors(i)
            for ind, off in zip(indices_, offsets_):
                ind_ = [ind]+list(off)
                ind_ = tuple(ind_)
                total_second_neighbor.update([ind_])
        for i in total_second_neighbor:
            if int(i[0]) in surface_atoms:
                second_neighbor_surf[slab[i[0]].symbol] +=1
            if int(i[0]) in subsurface_atoms:
                second_neighbor_sub[slab[i[0]].symbol] +=1
    
        return first_neighbor_surf, second_neighbor_surf, second_neighbor_sub 

    if len(first_neighbor) == 2:
        total_second_neighbor = []
        total_first_neighbor = []
        for i in first_neighbor:
            total_first_neighbor.append((i,0,0,0))
            indices_, offsets_ = n1.get_neighbors(i)
            for ind, off in zip(indices_, offsets_):
                ind_ = [ind] + list(off)
                ind_ = tuple(ind_)
                total_second_neighbor.append([ind_])
        sn = [i[0] for i in total_second_neighbor]
        to_remove = set(sn).intersection(set(total_first_neighbor))
        sn = [i for i in sn if i not in to_remove or type(i) != tuple]
        counted_twice = list(set([ele for ele in sn if sn.count(ele) >1]))
        sn = set(sn)
        for i in sn:
            if int(i[0]) in surface_atoms:
                if i in counted_twice:
                    fourth_neighbor_surf[slab[i[0]].symbol] +=1
                elif slab.get_distance(16, i[0]) == mx or slab.get_distance(16, i[0]) == mx2:
                    second_neighbor_surf[slab[i[0]].symbol] +=1
                else:
                    sixth_neighbor_surf[slab[i[0]].symbol] +=1
            if int(i[0]) in subsurface_atoms:
                if i in counted_twice:
                    fifth_neighbor_sub[slab[i[0]].symbol] +=1
                else:
                    second_neighbor_sub[slab[i[0]].symbol] +=1

        return first_neighbor_surf, second_neighbor_surf, second_neighbor_sub, fourth_neighbor_surf, fifth_neighbor_sub, sixth_neighbor_surf

    if len(first_neighbor) == 3:
        total_second_neighbor = []
        total_first_neighbor = []
        for i in first_neighbor:
            total_first_neighbor.append((i,0,0,0))
            indices_, offsets_ = n1.get_neighbors(i)
            for ind, off in zip(indices_, offsets_):
                ind_ = [ind] + list(off)
                ind_ = tuple(ind_)
                total_second_neighbor.append([ind_])
        sn = [i[0] for i in total_second_neighbor]
        to_remove = set(sn).intersection(set(total_first_neighbor))
        sn = [i for i in sn if i not in to_remove or type(i) != tuple]
        counted_twice = list(set([ele for ele in sn if sn.count(ele) >1]))
        sn = set(sn)
        for i in sn:
            if int(i[0]) in surface_atoms:
                if i in counted_twice:
                    fourth_neighbor_surf[slab[i[0]].symbol] +=1
                else:
                    second_neighbor_surf[slab[i[0]].symbol] +=1
            if int(i[0]) in subsurface_atoms:
                if i in counted_twice:
                    fifth_neighbor_sub[slab[i[0]].symbol] +=1
                else:
                    second_neighbor_sub[slab[i[0]].symbol] +=1

        return first_neighbor_surf, second_neighbor_surf, second_neighbor_sub, fourth_neighbor_surf, fifth_neighbor_sub

def get_first_neighbor_indices(slab, adsorbate_specie='H'):
    ads_atoms = get_adsorbates(slab)
    print(ads_atoms)
    direct_ads_atom = [ads_ for ads_ in ads_atoms if ads_.symbol in adsorbate_specie]
    print(direct_ads_atom)
    pos = direct_ads_atom[0].position

    nat_cutoff = ase_neighborlist.natural_cutoffs(slab)
    n1 = ase_neighborlist.NeighborList(nat_cutoff, self_interaction=False, bothways=True)
    n1.update(slab)
    indices, offsets = n1.get_neighbors(ads_atoms[0].index)
    indices = np.delete(indices, np.where((indices >16)))
    return indices
