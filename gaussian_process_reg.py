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

# from sklearn.gaussian_process import GaussianProcess

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, ConstantKernel as C

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


def GPR(v_file, del_file, rho_file, N, args=None, num_points=4, diff=7):
    return GPR_in(del_file, N, num_points, diff), GPR_in(rho_file, N, num_points, diff)


def GPR_in(filename, N, num_pts, diff=7):
    
    world, shock_data = read_shock_data(filename)

    Sr_val, Sr_index, tru_vals, err, NN = organize_data(shock_data)
    
    
    '''set up input data'''
    
    X = list(Sr_index.keys())
    X = [Sr for Sr in X if abs(Sr[0])<.11]
    # X_err = [err[Sr_index[Sr]] for Sr in X]
    
    
    '''choose points to subsample'''
    
    # these numbers can be as big as the corresponding ones in NN
    samp_N = np.zeros(tru_vals.shape)
    

    Sr_weights = [(5+int(10+9*math.sin(2*math.pi*j/num_pts)), int(10+9*math.cos(2*math.pi*j/num_pts))) for j in range(num_pts)]

    
    # S_weight_list = range(5,26,2)
    # r_weight_list = range(0,21,2)
    # Sr_weights = list(it.product(S_weight_list, r_weight_list))    
    
    for S_ind, r_ind in Sr_weights:
        samp_N[S_ind,r_ind] += int(N/(3*len(Sr_weights))   *1) 

    
    print(3*np.sum(samp_N))
    
    samp_X = [Sr for Sr in X if samp_N[Sr_index[Sr]]>0]
    
    noise = np.random.standard_normal(tru_vals.size)
    noise = noise.reshape(tru_vals.shape)
    samp_vals, samp_err = downsample(tru_vals, err, NN, Sr_index, samp_N, noise)
    
    samp_X_err = np.array([samp_err[Sr_index[Sr]] for Sr in X if samp_N[Sr_index[Sr]]>0])
    
    
    samp_true_y = [samp_vals[Sr_index[Sr]] for Sr in X if samp_N[Sr_index[Sr]]>0]
    samp_weights = [samp_N[Sr_index[Sr]] for Sr in X if samp_N[Sr_index[Sr]]>0]
    
    '''fit the model'''
    
    polynomial_features= PolynomialFeatures(degree=2)
    X_poly = polynomial_features.fit_transform(X)
    X_poly = X_poly[:,:num_pts-diff]

    samp_X_poly = [X_poly[j] for j in range(len(X)) if samp_N[Sr_index[X[j]]]>0]
    
    
    model_l = LinearRegression()
    model_l.fit(samp_X_poly, samp_true_y, sample_weight=samp_weights)
    
    samp_true_y = [samp_vals[Sr_index[samp_X[j]]] - model_l.predict([samp_X_poly[j]]) for j in range(len(samp_X))]
    # samp_true_y = [samp_vals[Sr_index[Sr]] - model_l.predict([Sr]) for Sr in X if samp_N[Sr_index[Sr]]>0]
    
    
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)) * RBF(10, (1e-2, 1e2))
    # kernel = C(1.0, (1e-3, 1e3)) * RationalQuadratic(10, 1)
    model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9,
                                     normalize_y=True, alpha=samp_X_err)
    samp_X = [np.array([Sr[0], Sr[1]]) for Sr in samp_X]
    model.fit(samp_X, samp_true_y)
    
    '''see how the model does'''
    
    X = [Sr for Sr in X if abs(Sr[0])<.11]
    true_y = [tru_vals[Sr_index[Sr]] for Sr in X]
    
    pred_y = model.predict(X).reshape((-1,)) + model_l.predict(X_poly).reshape((-1,))
    # pred_y = model.predict(X).reshape((-1,)) + model_l.predict(X).reshape((-1,))
    mse = mean_squared_error(true_y, pred_y)
    print(mse)
    
    predicts = zip(X, pred_y)
    model_err = np.zeros(tru_vals.shape)
    for Sr, V in predicts:
        model_err[Sr_index[Sr]] = tru_vals[Sr_index[Sr]] - V
    model_err = np.abs(model_err/tru_vals)
    model_err = model_err[5:26,:]    
    
    '''graph it'''
    
    # plt.imshow(model_err, cmap='magma')
    # plt.colorbar()
    # plt.show()
    # sns.kdeplot(model_err.reshape((-1)), shade=True)
    # plt.show()

    return model_err














