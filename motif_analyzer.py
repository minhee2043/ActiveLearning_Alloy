from ase import Atoms
import csv
import pandas as pd
import matplotlib.pyplot as plt
import sys
import ase
from ase.io import read
import pickle
from ase.calculators.vasp import Vasp2
import os
from ase.io import read
from helperMethods import count_atoms, sortMetals
import itertools as it
import numpy as np


class Slab(object):
    '''slab class'''

    def __init__(self, atoms=None):
        self.atoms = atoms              # Atoms object
        self.slab3x3 = None             # atoms repeated 3x3x1
        self.site = None                # adsorption site ('onTop','hcp','fcc')

    def get_site(self):
        '''return the id and type of metal in sublayer, return None if no metal
        and update the instance variable 'site' with either 'hcp' or 'fcc' '''
        ensSym, ensIds = self.closest(
            layer=1, start=1, stop=3)  # define ensemble
        slab3x3 = self.slab3x3

        # (3x2), ensemble x,y positions
        ensPos = slab3x3.get_positions()[ensIds][:, 0:2]
        # the ensemble triangle
        triangle = Triangle(ensPos[0], ensPos[1], ensPos[2])

        # sublayer atom ids
        subIds = [atom.index for atom in slab3x3 if atom.tag == 2]
        # (36x2), sublayer x,y positions
        subPos = slab3x3.get_positions()[subIds][:, 0:2]

        count, hcp = 0, False
        for i, pos in enumerate(subPos):
            if triangle.within(pos):  # if submetal is within triangle
                count += 1
                hcp = True  # metal underneath, so hcp hole detected
                subId = subIds[i]
                symbols = slab3x3.get_chemical_symbols()
                subMetal = symbols[subId]
        if count > 1:
            print('Warning: %i metals detected in hcp hole' % count)
        if hcp:
            self.site = 'hcp'
            return subMetal, subId
        else:
            self.site = 'fcc'
            return None, None

    def adsorbate_id(self):
        '''return id for the adsorbing atom'''

        # number of times to extend slab in x and y direction
        n = 3

        # extend slab
        slab3x3 = self.atoms.repeat((n, n, 1))
        self.slab3x3 = slab3x3

        # ids of adsorbate atoms on 3x3 slab
        adsIds = np.array([atom.index for atom in slab3x3 if atom.tag ==
                           0])

        # number of atoms in adsorbate (e.g. two atoms in OH)
        nAdsAtoms = int(len(adsIds) / (n*n))

        # indices of ids of adsorbing atoms
        keepIds = range(0, len(adsIds), nAdsAtoms)
        
        # ids of adsorbing atoms
        adsIds = adsIds[keepIds]

        # reshape to square matrix
        adsIds = adsIds.reshape((n, n))

        # return the id of the central adsorbing atom
        nHalf = int(n/2)
        return adsIds[nHalf, nHalf]

    def closest(self, layer, start, stop):
        '''return a list of symbols and ids for atoms closest to the adsorbate
        in the layer given (1: surface, 2: subsurface) in the interval [start,stop],
        e.g. for the 3 closest: [1,3]; the 9 next closest: [4,12]'''
        # atom ids
        adsId = self.adsorbate_id()
        slab3x3 = self.slab3x3
        layerIds = [atom.index for atom in slab3x3 if atom.tag == layer]
        # dist from adsorbate for the layer
        layerDist = slab3x3.get_distances(adsId, layerIds)
        indexedDist = [[layerIds[i], layerDist[i]]
                       for i in range(len(layerDist))]  # unite lists
        sortedDist = sorted(indexedDist, key=lambda x: x[1])[
            start-1:stop]  # sort atom ids by distance
        ids = []  # ids of interest
        for i, dist in sortedDist:
            ids.append(i)  # make a separate sorted list of ids

        # atom symbols
        symbols = slab3x3.get_chemical_symbols()
        # convert into array to extract list of indices given
        symbols = np.array(symbols)
        zoneSym = list(symbols[ids])  # zone symbols of interest
        indexedSym = [[zoneSym[i], ids[i]] for i in range(len(ids))]
        # sort metals by atom id
        sortedSym = sorted(indexedSym, key=lambda x: x[1])
        zoneSym = []
        ids = []
        for metal, atomId in sortedSym:
            zoneSym.append(metal)
            ids.append(atomId)
        return zoneSym, ids

    def ensemble(self, onTop=False):
        '''return ensemble atom symbols
        onTop:  if True, return the symbol closest to adsorbate, else the 3 closest'''
        if onTop:
            return self.closest(layer=1, start=1, stop=1)[0]
        else:
            return self.closest(layer=1, start=1, stop=3)[0]

    def surface(self, onTop=False):
        '''return the neighbour surface symbols of the ensemble
        onTop:  if True, return the 6 symbols closest, else the 9 closest including double
        counting to 12 symbols'''
        if onTop:
            return (self.closest(layer=1, start=2, stop=7)[0] +
                   self.closest(layer=1, start=8, stop=10)[0])
        else:
            return (self.closest(layer=1, start=4, stop=6)[0]*2 +
                    self.closest(layer=1, start=7, stop=12)[0])

    def subsurface(self, onTop=False):
        '''return the symbols for the nearest subsurface neighbours of the ensemble
        onTop:  if True, return the 3 symbols closest in subsurface, else 7 (hcp) or 6 (fcc)'''
        if onTop:
            return (self.closest(layer=2, start=1, stop=3)[0] +
                    self.closest(layer=2, start=4, stop=6)[0])

        if self.site == None:
            self.get_site()  # update the instance variable site with 'hcp' or 'fcc'

        if self.site == 'fcc':
            return (self.closest(layer=2, start=1, stop=3)[0]*2 +
                    self.closest(layer=2, start=4, stop=6)[0])
        elif self.site == 'hcp':
            return (self.closest(layer=2, start=1, stop=1)[0]*3 +
                    self.closest(layer=2, start=2, stop=7)[0])
                    else:
            raise ValueError('the site was not classified')

    def surface_near(self, onTop=False):
        '''return nearest surface neighbour symbols of the ensemble
        onTop:  if True, return the 6 symbols closest, else the 3 closest'''
        if onTop:
            return self.closest(layer=1, start=2, stop=7)[0]
        else:
            return self.closest(layer=1, start=4, stop=6)[0]

    def surface_far(self):  
        '''return the symbols for the 3 (fcc),6 furthest surface neighbours of the ensemble'''
        if onTop:
            return self.closest(layer=1, start=8, stop=10)[0]
        else:
            return self.closest(layer=1, start=7, stop=12)[0]

    def subsurface_near(self, onTop=False):
        '''return the symbols for the nearest subsurface neighbours of the ensemble
        onTop:  if True, return the 3 symbols closest in subsurface, else 1 (hcp) or 3 (fcc)'''

        if onTop:
            return self.closest(layer=2, start=1, stop=3)[0]
        if self.site == None:
            self.get_site()  # update the instance variable site with 'hcp' or 'fcc'

        if self.site == 'fcc':
            return self.closest(layer=2, start=1, stop=3)[0]
        elif self.site == 'hcp':
            return self.closest(layer=2, start=1, stop=1)[0]
        else:
            raise ValueError('the site was not classified')

    def subsurface_far(self):  
        '''return the symbols for the 6 (hcp) or 3 (fcc) furthest subsurface neighbours of
        the ensemble'''
        if onTop:
            return self.closest(layer=2, start=4, stop=6)[0]
        if self.site == None:
            self.get_site()  # update the instance variable site with 'hcp' or 'fcc'

        if self.site == 'hcp':
            return self.closest(layer=2, start=2, stop=7)[0]
        elif self.site == 'fcc':
            return self.closest(layer=2, start=4, stop=6)[0]
        else:
            raise ValueError('the hollow site is neither fcc nor hcp')

    def distorted(self, threshold=1.10):
        '''return True if a surface atom has distorted more than the threshold in the z direction,
        False otherwise
        threshold:      fraction slab height after vs. before allowance, eg. 1.10 allows for a 10%
                                increase in the slab height'''
        
        atoms = self.atoms

        dist = atoms.get_distance(0, 1)  # 2 bottom layer atoms distance
        # initial layer separation distance
        layerDistInit = dist*np.sqrt(2.)/np.sqrt(3.)
        heightInit = 3*layerDistInit  # initial slab height
        heightLim = heightInit*threshold  # max allowed slab height after relaxation

        surIds, botIds = [], []
        for atom in atoms:
            if atom.tag == 1:
                surIds.append(atom.index)  # surface atom ids
            if atom.tag == 4:
                botIds.append(atom.index)  # bottom layer atom ids

        zsSur = atoms.get_positions()[surIds, 2]
        # all equal to the vacuum height, eg. 7.5
        zsBot = atoms.get_positions()[botIds, 2]
        zs = zsSur-zsBot
        maxZ = max(zs)

        if maxZ > heightLim:
            return True  # slab is distorted beyond the threshold
        else:
            return False  # slab is not distorted beyond the threshold

    def onTop(self, onTopDist=0.70):
        '''return True if x,y distance between adsorbate and closest surface metal is below
        threshold and False otherwise'''
        adsId = self.adsorbate_id()  # adsorbate id
        metalId = self.closest(1, 1, 1)[1]  # closest surface metal id
        [dist] = self.slab3x3.get_distance(adsId, metalId, vector=True)
        xyDist = np.sqrt(dist[0]**2 + dist[1]**2)

        if xyDist < onTopDist:
            return True
        else:
            return False

    def features(self, metals, onTop=False, zones=['ens', 'sf', 'ssf','sn','ssn']):
        '''return a list of counts for each metal in each zone in the slab, ensembles are given
        a 1 for the type of ensemble present
        metals:         list of reference metals to do the counting according to
        onTop:          set True if modelling an on-top adsorption
        zones:          list of zones to consider ['ens','sn','ssn','sf','ssf']
                                ens:    ensemble
                                {s  :   surface
                                ss :    subsurface  }
                                {sn :   surface near
                                ssn:    subsurface near
                                sf :    surface far
                                ssf:    subsurface far}'''

        feature = []

        for zone in zones:
            if zone == 'ens':
                if onTop:
                    ensSize = 1
                    siteSym = self.ensemble(onTop)
                    siteCount = count_atoms(siteSym,metals)*3 
                    feature+=siteCount
                else:
                    ensSize = 3
                    siteSym = self.ensemble(onTop)
                    siteCount = count_atoms(siteSym,metals)
                    feature+=siteCount
            elif zone == 's':
                                sSym = self.surface(onTop)  # symbols including double counting
                sCount = count_atoms(sSym, metals)
                feature += sCount
            elif zone == 'ss':
                # symbols including double counting
                ssSym = self.subsurface(onTop)
                ssCount = count_atoms(ssSym, metals)
                feature += ssCount
            elif zone == 'sn':
                snSym = self.surface_near(onTop)
                snCount = count_atoms(snSym, metals)
                feature += snCount
            elif zone == 'ssn':
                ssnSym = self.subsurface_near(onTop)
                ssnCount = count_atoms(ssnSym, metals)
                feature += ssnCount
            elif zone == 'sf':
                sfSym = self.surface_far()
                sfCount = count_atoms(sfSym, metals)
                feature += sfCount
            elif zone == 'ssf':
                ssfSym = self.subsurface_far()
                ssfCount = count_atoms(ssfSym, metals)
                feature += ssfCount
        return feature

# --- helper triangle class, to classify hollow sites --- #


class Triangle(object):
    '''triangle class'''

    def __init__(self, p1, p2, p3):
        self.p1 = np.array(p1)
        self.p2 = np.array(p2)
        self.p3 = np.array(p3)

    def within(self, p):
        '''return True if the point p is within the triangle, False if not.
        Implementation from http://mathworld.wolfram.com/TriangleInterior.html'''
        v = np.array(p)
        v0 = self.p1
        v1 = self.p2-v0
        v2 = self.p3-v0
        detvv2 = np.cross(v, v2)
        detv0v2 = np.cross(v0, v2)
        detv1v2 = np.cross(v1, v2)
        detvv1 = np.cross(v, v1)
        detv0v1 = np.cross(v0, v1)
        a = (detvv2-detv0v2)/detv1v2
        b = -(detvv1-detv0v1)/detv1v2
        if a > 0 and b > 0 and a+b < 1:
            return True
        else:
            return False
                                                                                                                                          154,1         22%
