import sys
import os
from scipy.stats import norm
import csv
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.gaussian_process as gp
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import math
import pandas as pd
from pandas import DataFrame

#########implementation of the "Dealing with categorical and integer-valued variables in bayesian optimization with Gaussian Processes" authored by Garrido-Merchan & Hernandez-Lobato
from sklearn.gaussian_process.kernels import *
#StationaryKernelMixin, NormalizedKernelMixin, Kernel, Hyperparameter
import bisect
from joblib import Parallel, delayed

def process_file(i):
    x_val, y_val = [], []
    feat_multi=[]
    with open(f'../groupcount_{i}.csv', 'r') as handle:
        for line in handle.readlines()[0:]:

            f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,mult=line[:].split(',')[:28]
            x_val.append([f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25])
            feat_multi.append([int(f1),int(f2),int(f3),int(f4),int(f5),int(f6),int(f7),int(f8),int(f9),int(f10),int(f11),int(f12),int(f13),int(f14),int(f15),int(f16),int(f17),int(f18),int(f19),int(f20),int(f21),int(f22),int(f23),int(f24),int(f25),int(f26),int(f27),int(mult)])

    x_val = np.array(x_val)
    X_val = np.array(x_val, dtype=int)

    y_val, dev = model.predict(X_val, return_std=True)
    y_val = np.array(y_val)
    dev = np.array(dev)
    imp = -abs(y_val+0.27-0.01)
    Z = imp/dev
    uncertainty = imp * norm.cdf(Z) + dev * norm.pdf(Z)

    output=np.c_[feat_multi,y_val,dev,uncertainty]
    np.savetxt(f'GPR_batch20/GPR_group{i}.csv',output, fmt=['%d']*28+['%.5f']*3,delimiter=',')


def _check_length_scale(X, length_scale):
    length_scale = np.squeeze(length_scale).astype(float)
    if np.ndim(length_scale) > 1:
        raise ValueError("length_scale cannot be of dimension greater than 1")
    if np.ndim(length_scale) == 1 and X.shape[1] != length_scale.shape[0]:
        raise ValueError(
            "Anisotropic kernel must have the same number of "
            "dimensions as data (%d!=%d)" % (length_scale.shape[0], X.shape[1])
        )
    return length_scale


class RBF_int(StationaryKernelMixin, NormalizedKernelMixin, Kernel):

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic:
            return Hyperparameter(
                "length_scale",
                "numeric",
                self.length_scale_bounds,
                len(self.length_scale),
            )
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)
      
    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        Xfilter = np.around(X)
        X = Xfilter
        length_scale = _check_length_scale(X, self.length_scale)
        if Y is None:
            dists = pdist(X / length_scale, metric="sqeuclidean")

            K = np.exp(-0.5 * dists)
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            dists = cdist(X / length_scale, Y / length_scale, metric="sqeuclidean")
            K = np.exp(-0.5 * dists)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, np.empty((X.shape[0], X.shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                K_gradient = (K * squareform(dists))[:, :, np.newaxis]
                return K, K_gradient
            elif self.anisotropic:
                # We need to recompute the pairwise dimension-wise distances
                K_gradient = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 / (
                    length_scale ** 2
                )
                K_gradient *= K[..., np.newaxis]
                return K, K_gradient
        else:
            return K

    def __repr__(self):
        if self.anisotropic:
            return "{0}(length_scale=[{1}])".format(
                self.__class__.__name__,
                ", ".join(map("{0:.3g}".format, self.length_scale)),
            )
        else:  # isotropic
            return "{0}(length_scale={1:.3g})".format(
                self.__class__.__name__, np.ravel(self.length_scale)[0])





x_all, y_all = [], []
with open('DFT/DFT_all.csv', 'r') as handle:
  for line in handle.readlines()[0:]:
    f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,energy=line[:].split(',')[:28]
    x_all.append([f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27])
    y_all.append(float(energy))
x_all = np.array(x_all)
X_all = np.array(x_all, dtype=int) #  convert using numpy
y_all = np.array(y_all)

x_train, x_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.25, random_state=42)

kernel = gp.kernels.ConstantKernel(5.0, (1e-1, 1e3)) *RBF_int(length_scale=0.25*np.ones((27,)),
            length_scale_bounds=(1.0e-1, 1.0e3))
#print(kernel)
model = gp.GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=10,alpha=0.1, normalize_y=True)
model.fit(x_train, y_train)

params = model.kernel_.get_params()
#print(params)
y_pred=model.predict(x_test, return_std=False)
MAE=mean_absolute_error(y_test, y_pred)
#print(x_test)
#print(y_pred)
#print(y_test)
print(MAE)
print(mean_squared_error(y_test, y_pred)**0.5)


num_cores = 24  # Change this to the number of cores you want to use
Parallel(n_jobs=num_cores)(delayed(process_file)(i) for i in range(0,35))

xs,ys,mult=[],[],[]
with open('../possiblesite.csv', 'r') as handle:
        for line in handle.readlines()[0:]:
            f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,mult=line[:].split(',')[:28]
            xs.append([f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27])
            multt.append(int(mult))
xs=np.array(xs,dtype=int)

unc=[]
ys,dev =model.predict(xs,return_std=True)
ys=np.array(ys)
dev=np.array(dev)
imp = -abs(ys+0.27-0.01)
Z = imp/dev
unc = imp*norm.cdf(Z) +dev*norm.pdf(Z)

output = np.c_[xs,mult,ys,dev,unc]
np.savetxt('result/stable_GPR19.csv',output,fmt=['%d']*28+['%.5f']*3,delimiter=',')

def find_largest_rows(file_path):
    row_num=[]
    # Open the CSV file
    with open(file_path, 'r') as f:
        # Create a CSV reader object
        reader = csv.reader(f)
        # Create a list to store the rows
        rows = []
        # Iterate over the rows in the CSV
        for i, row in enumerate(reader):
            # Append the row and its index to the list
            rows.append((i, row))
        # Sort the list by the last column in descending order
        sorted_rows = sorted(rows, key=lambda x: float(x[1][-1]), reverse=True)
        top_50 =sorted_rows[:50]
        col = [float(row[1][-1]) for row in top_50]
        mean = sum(col)/len(col)
        print(mean)
        # Print the index of the first 50 rows
        for i in range(50):
            row_num.append(sorted_rows[i][0])
    return row_num

# Usage example
row_num= find_largest_rows('result/stable_GPR19.csv')

with open('../index_metal.csv','r') as f:
    indtomet=f.readlines()

with open('batch20_metal_stable.csv','w') as fout:
    for num in row_num:
        fout.write(indtomet[num])
      
