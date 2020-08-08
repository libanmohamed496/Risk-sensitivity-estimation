    # -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 13:16:03 2020

@author: liban
"""


import itertools as it
import functools as ft
import operator as op

import math
import numpy as np
import scipy as sp
import random
import numba

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

import matplotlib.pyplot as plt
import seaborn as sns

import gmab_model
from gmab_model import set_world, make_future_gen
from gmab_model import vec_rollforward, vec_GMAB, vec_GMAB_ratchet
from gmab_model import delta, rho
from gmab_model import value_by_rep, value_by_error, value_by_precision
from gmab_model import shock_value, write_shock_data, read_shock_data, organize_data


def downsample(tru_vals, err, NN, Sr_index, samp_N, noise):
    
    samp_vals = np.zeros(tru_vals.shape)
    samp_err = np.zeros(tru_vals.shape)
    
    # technically biased, but at least it doesn't blow up:
    samp_err = err*np.power((NN-samp_N+1)/((samp_N+1)), .5)
    samp_vals = tru_vals + samp_err*noise
         
    return samp_vals, samp_err

world, shock_data = read_shock_data('sample6/sample6_delta_.csv')

Sr_val, Sr_index, tru_vals, err, NN = organize_data(shock_data)


'''set up input data'''

X = list(Sr_index.keys())
true_y = [tru_vals[Sr_index[Sr]] for Sr in X]
# X_err = [err[Sr_index[Sr]] for Sr in X]
weights = [1 for Sr in X]


'''choose points to subsample'''


# these numbers can be as big as the corresponding ones in NN
samp_N = np.zeros(tru_vals.shape)

samp_Sr = [(0,0),(.01,0),(.02,0),(.03,0),
        (.05,0),(.1,0),(-.01,0),
        (-.02,0),(-.03,0),(-.05,0),(-.1,0)]

for Sr in samp_Sr:
    samp_N[Sr_index[Sr]] = 100

samp_X = [Sr for Sr in X if samp_N[Sr_index[Sr]]>0]
samp_X_err = np.array([err[Sr_index[Sr]] for Sr in X if samp_N[Sr_index[Sr]]>0])

noise = np.random.standard_normal(tru_vals.size)
noise = noise.reshape(tru_vals.shape)
samp_vals, samp_err = downsample(tru_vals, err, NN, Sr_index, samp_N, noise)

samp_true_y = [samp_vals[Sr_index[Sr]] for Sr in X if samp_N[Sr_index[Sr]]>0]
samp_weights = [samp_N[Sr_index[Sr]] for Sr in X if samp_N[Sr_index[Sr]]>0]

'''fit the model'''

model = LinearRegression()
model.fit(samp_X, samp_true_y, sample_weight = samp_weights)


'''see how the model does'''

pred_y = model.predict(X)
mse = mean_squared_error(true_y, pred_y, sample_weight = weights)
print(mse)

predicts = zip(X, pred_y)
model_err = np.zeros(tru_vals.shape)
for Sr, V in predicts:
    model_err[Sr_index[Sr]] = V - tru_vals[Sr_index[Sr]]
model_err = np.abs(model_err)

'''graph it'''

plt.imshow(model_err, cmap='magma')
plt.colorbar()
plt.show()
sns.kdeplot(model_err.reshape((-1)), shade=True)
plt.show()


















