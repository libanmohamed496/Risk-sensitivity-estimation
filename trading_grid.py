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

from scipy.interpolate import CubicSpline
from scipy.interpolate import UnivariateSpline

import matplotlib.pyplot as plt
import seaborn as sns

import gmab_model
from gmab_model import set_world, make_future_gen
from gmab_model import vec_rollforward, vec_GMAB, vec_GMAB_ratchet
from gmab_model import delta, rho
from gmab_model import value_by_rep, value_by_error, value_by_precision
from gmab_model import shock_value, write_shock_data, read_shock_data


def get_data(filename):
    world, shock_dat = read_shock_data(filename)
    
    S_vals = list(set(map(lambda tup : tup[0][0], shock_dat)))
    S_vals.sort()
    
    r_vals = list(set(map(lambda tup : tup[0][1], shock_dat)))
    r_vals.sort()
    
    S_index = list(zip(S_vals,it.count()))
    r_index = list(zip(r_vals,it.count()))
    Sr_index = dict(map(lambda tup : ((tup[0][0], tup[1][0]), (tup[0][1], tup[1][1])),
                        it.product(S_index, r_index)))
    
    tru_vals = np.zeros((len(S_vals),len(r_vals)))
    err = np.zeros((len(S_vals),len(r_vals)))
    NN = np.zeros((len(S_vals),len(r_vals)))
    for dat in shock_dat:
        S = dat[0][0]
        r = dat[0][1]
        tru_vals[Sr_index[(S,r)]] = dat[2][0]
        err[Sr_index[(S,r)]] = dat[2][1]                     
        NN[Sr_index[(S,r)]] = dat[2][2]
                    
    return world, S_vals, r_vals, Sr_index, tru_vals, err, NN
  
def downsample(tru_vals, err, NN, Sr_index, samp_N, noise):
    
    samp_vals = np.zeros(tru_vals.shape)
    samp_err = np.zeros(tru_vals.shape)
    
    # technically biased, but at least it doesn't blow up:
    samp_err = err*np.power((NN-samp_N+1)/((samp_N+1)), .5)
    samp_vals = tru_vals + samp_err*noise
         
    return samp_vals, samp_err

def TGrid(v_file, del_file, rho_file, N, GMAB_args):
    world, S_vals, r_vals, Sr_index, tru_val, val_err, val_NN = get_data(v_file)
    world, S_vals, r_vals, Sr_index, tru_delta, delta_err, delta_NN = get_data(del_file)
    world, S_vals, r_vals, Sr_index, tru_rho, rho_err, rho_NN = get_data(rho_file)
    
    world = world + (None,)
    '''set up input data'''
    
    X = list(Sr_index.keys())
    
    for Sr in X:
        pass
        tru_delta[Sr_index[Sr]] /= 1+Sr[0]
    
    
    '''choose points to subsample'''
    
    # these numbers can be as big as the corresponding ones in NN
    samp_N = np.zeros(tru_delta.shape)
    
    samp_Sr = [(0,0),(.01,0),(.02,0),(.03,0),
            (.05,0),(.1,0),(-.01,0),
            (-.02,0),(-.03,0),(-.05,0),(-.1,0), (.15,0), (-.15,0)]
    
    num_points = 13
    num_MC = int((N-6)/(num_points-1))
    # samp_Sr = [(int(10*math.cos(math.pi*j/(num_points-1)))/100,0) for j in range(num_points)]
    
    for Sr in samp_Sr:
        samp_N[Sr_index[Sr]] = 2000
    
    samp_lia = np.zeros(tru_delta.shape)
    lia_err = np.zeros(tru_delta.shape)
    lia_NN = np.zeros(tru_delta.shape)
    
    samp_r = np.zeros(tru_delta.shape)
    r_err = np.zeros(tru_delta.shape)
    r_NN = np.zeros(tru_delta.shape)
    
    samp_dr = np.zeros(tru_delta.shape)
    dr_err = np.zeros(tru_delta.shape)
    dr_NN = np.zeros(tru_delta.shape)
    
    samp_rr = np.zeros(tru_delta.shape)
    rr_err = np.zeros(tru_delta.shape)
    rr_NN = np.zeros(tru_delta.shape)
    
    '''get MC estimates at each point'''
    
    lia_shocks = shock_value(world, samp_Sr, value_by_rep, num_MC, vec_GMAB_ratchet, *GMAB_args)
    r_shocks = shock_value(world, [(0,0)], value_by_rep, num_MC, rho, .001, vec_GMAB_ratchet, *GMAB_args)
    dr_shocks = shock_value(world, [(0,0)], value_by_rep, num_MC, rho, .001, delta, .001, vec_GMAB_ratchet, *GMAB_args)
    rr_shocks = shock_value(world, [(0,0)], value_by_rep, num_MC, rho, .001, rho, .001, vec_GMAB_ratchet, *GMAB_args)
    
    for shock in lia_shocks:
        Sr = shock[0]
        val_err_N = shock[2]
        samp_lia[Sr_index[Sr]] = val_err_N[0]
        lia_err[Sr_index[Sr]] = val_err_N[1]
        lia_NN[Sr_index[Sr]] = val_err_N[2]
    
    for shock in r_shocks:
        Sr = shock[0]
        val_err_N = shock[2]
        samp_r[Sr_index[Sr]] = val_err_N[0]
        r_err[Sr_index[Sr]] = val_err_N[1]
        r_NN[Sr_index[Sr]] = val_err_N[2]
        
    for shock in dr_shocks:
        Sr = shock[0]
        val_err_N = shock[2]
        samp_dr[Sr_index[Sr]] = val_err_N[0]
        dr_err[Sr_index[Sr]] = val_err_N[1]
        dr_NN[Sr_index[Sr]] = val_err_N[2]
        
    for shock in rr_shocks:
        Sr = shock[0]
        val_err_N = shock[2]
        samp_rr[Sr_index[Sr]] = val_err_N[0]
        rr_err[Sr_index[Sr]] = val_err_N[1]
        rr_NN[Sr_index[Sr]] = val_err_N[2]
    
    samp_X = [Sr[0] for Sr in X if samp_N[Sr_index[Sr]]>0]
    samp_X_err = [lia_err[Sr_index[Sr]] for Sr in X if samp_N[Sr_index[Sr]]>0]
    
    # noise = np.random.standard_normal(tru_delta.size)
    # noise = noise.reshape(tru_delta.shape)
    # samp_vals, samp_err = downsample(tru_rho, rho_err, rho_NN, Sr_index, samp_N, noise)
    
    samp_true_y = [samp_lia[Sr_index[Sr]] for Sr in X if samp_N[Sr_index[Sr]]>0]
    samp_weights = [samp_N[Sr_index[Sr]] for Sr in X if samp_N[Sr_index[Sr]]>0]
    
    '''fit the model'''
    
    # model = LinearRegression()
    # model.fit(samp_X, samp_true_y, sample_weight = samp_weights)
    
    '''''''''''''''gota fils xhit onsliseatlyo pay attemtnoi to me'''
    model = UnivariateSpline(samp_X, samp_true_y, w=1/np.array(samp_X_err))
    model_prime = model.derivative()
    
    
    # cross shocks: (0,4.8e-05),(0,-4.8e-05),(-.01,-4.8e-05)
    
    '''see how the model does'''
    X = [Sr for Sr in X if abs(Sr[0])<.11]
    true_y_d = [tru_delta[Sr_index[Sr]] for Sr in X]
    pred_y_d = [model_prime(Sr[0])/(1+Sr[0]) + Sr[1]*samp_dr[Sr_index[(0,0)]]/(1+Sr[0]) for Sr in X]
    
    true_y_r = [tru_rho[Sr_index[Sr]] for Sr in X]
    pred_y_r = [samp_r[Sr_index[(0,0)]] + Sr[1]*samp_rr[Sr_index[(0,0)]] for Sr in X]
    
    mse_d = mean_squared_error(true_y_d, pred_y_d)
    mse_r = mean_squared_error(true_y_r, pred_y_r)
    print(mse_d)
    print(mse_r)
    
    predicts_d = zip(X, pred_y_d)
    model_err_d = np.zeros(tru_delta.shape)
    for Sr, V in predicts_d:
        model_err_d[Sr_index[Sr]] = V - tru_delta[Sr_index[Sr]]/(1+Sr[0])
    model_err_d = np.abs(model_err_d/tru_delta)
    
    predicts_r = zip(X, pred_y_r)
    model_err_r = np.zeros(tru_delta.shape)
    for Sr, V in predicts_r:
        model_err_r[Sr_index[Sr]] = V - tru_rho[Sr_index[Sr]]
    model_err_r = np.abs(model_err_r/tru_rho)
    
    model_err_d = model_err_d[5:26,:]
    model_err_r = model_err_r[5:26,:]
    
    return model_err_d, model_err_r

# '''graph it'''

# plt.imshow(model_err_d, cmap='magma')
# plt.title('Error Location Distribution')
# plt.ylabel('S from 90% to 110%')
# plt.xlabel('-.00005 < r < .00005')
# plt.colorbar()
# plt.show()

# sns.kdeplot(model_err_d.reshape((-1)), shade=True)
# plt.title('Error Size Distribution')
# plt.ylabel('Error Density')
# plt.xlabel('Error Size')
# plt.show()

# plt.imshow(model_err_r, cmap='magma')
# plt.title('Error Location Distribution')
# plt.ylabel('S from 90% to 110%')
# plt.xlabel('-.00005 < r < .00005')
# plt.colorbar()
# plt.show()

# sns.kdeplot(model_err_r.reshape((-1)), shade=True)
# plt.title('Error Size Distribution')
# plt.ylabel('Error Density')
# plt.xlabel('Error Size')
# plt.show()


# arr.append(mse_d)













