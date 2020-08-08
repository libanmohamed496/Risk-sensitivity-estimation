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



def cube_spline(v_file, del_file, rho_file, N, args, num_points=10):

    world, S_vals, r_vals, Sr_index, tru_delta, delta_err, delta_NN = get_data(del_file)
    world, S_vals, r_vals, Sr_index, tru_rho, rho_err, rho_NN = get_data(rho_file)

    world = world + (None,)
    '''set up input data'''
    
    X = list(Sr_index.keys())
    weights = [1 for Sr in X]
    
    
    '''choose points to subsample'''
    
    num_points = 10
    
    # these numbers can be as big as the corresponding ones in NN
    samp_N = np.zeros(tru_delta.shape)
    
    # samp_Sr = [(0,0),(.01,0),(.02,0),(.03,0),
    #         (.05,0),(.1,0),(-.01,0),
    #         (-.02,0),(-.03,0),(-.05,0),(-.1,0)]
   
    samp_Sr = [(int(10*math.cos(math.pi*j/(num_points-1)))/100,0) for j in range(num_points)]
    # print(samp_Sr)
    
    for Sr in samp_Sr:
        samp_N[Sr_index[Sr]] = int(N/(5*len(samp_Sr)))
    print(5*np.sum(samp_N))
        
    samp_dd = np.zeros(tru_delta.shape)
    dd_err = np.zeros(tru_delta.shape)
    dd_NN = np.zeros(tru_delta.shape)
    
    samp_dr = np.zeros(tru_delta.shape)
    dr_err = np.zeros(tru_delta.shape)
    dr_NN = np.zeros(tru_delta.shape)
    
    samp_rr = np.zeros(tru_delta.shape)
    rr_err = np.zeros(tru_delta.shape)
    rr_NN = np.zeros(tru_delta.shape)
    
    dd_shocks = shock_value(world, samp_Sr, value_by_rep, 2, delta, .001, delta, .001, vec_GMAB_ratchet, *args)
    dr_shocks = shock_value(world, samp_Sr, value_by_rep, 2, rho, .001, delta, .001, vec_GMAB_ratchet, *args)
    rr_shocks = shock_value(world, samp_Sr, value_by_rep, 2, rho, .001, rho, .001, vec_GMAB_ratchet, *args)
    
    for shock in dd_shocks:
        Sr = shock[0]
        val_err_N = shock[2]
        samp_dd[Sr_index[Sr]] = val_err_N[0]
        dd_err[Sr_index[Sr]] = val_err_N[1]
        dd_NN[Sr_index[Sr]] = val_err_N[2]
        
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
    # samp_X.sort()
    
    noise = np.random.standard_normal(tru_delta.size)
    noise = noise.reshape(tru_delta.shape)
    samp_delta, samp_err_d = downsample(tru_delta, delta_err, delta_NN, Sr_index, samp_N, noise)
    samp_rho, samp_err_r = downsample(tru_rho, rho_err, rho_NN, Sr_index, samp_N, noise)
    
    samp_true_y_d = [samp_delta[Sr_index[Sr]] for Sr in X if samp_N[Sr_index[Sr]]>0]
    samp_true_y_r = [samp_rho[Sr_index[Sr]] for Sr in X if samp_N[Sr_index[Sr]]>0]
    samp_true_y_dr = [samp_dr[Sr_index[Sr]] for Sr in X if samp_N[Sr_index[Sr]]>0]
    samp_true_y_rr = [samp_rr[Sr_index[Sr]] for Sr in X if samp_N[Sr_index[Sr]]>0]
    samp_weights = [samp_N[Sr_index[Sr]] for Sr in X if samp_N[Sr_index[Sr]]>0]
    
    '''fit the model'''
    
    # model = LinearRegression()
    # model.fit(samp_X, samp_true_y, sample_weight = samp_weights)
    
    model_d = UnivariateSpline(samp_X, samp_true_y_d)
    model_r = UnivariateSpline(samp_X, samp_true_y_r)
    model_dr = UnivariateSpline(samp_X, samp_true_y_dr)
    model_rr = UnivariateSpline(samp_X, samp_true_y_rr)
    
    # cross shocks: (0,4.8e-05),(0,-4.8e-05),(-.01,-4.8e-05)
    
    '''see how the model does'''
    X = [Sr for Sr in X if abs(Sr[0])<.11]
    true_y_d = [tru_delta[Sr_index[Sr]] for Sr in X]
    true_y_r = [tru_rho[Sr_index[Sr]] for Sr in X]
    pred_y_d = [model_d(Sr[0])+Sr[1]*model_dr(Sr[0]) for Sr in X]
    pred_y_r = [model_r(Sr[0])+Sr[1]*model_rr(Sr[0]) for Sr in X]
    mse_d = mean_squared_error(true_y_d, pred_y_d)
    mse_r = mean_squared_error(true_y_r, pred_y_r)
    print(mse_d)
    
    predicts = zip(X, pred_y_d)
    model_err_d = np.zeros(tru_delta.shape)
    for Sr, V in predicts:
        model_err_d[Sr_index[Sr]] = V - tru_delta[Sr_index[Sr]]
        
    predicts = zip(X, pred_y_r)
    model_err_r = np.zeros(tru_delta.shape)
    for Sr, V in predicts:
        model_err_r[Sr_index[Sr]] = V - tru_rho[Sr_index[Sr]]
        
    model_err_d = np.abs(model_err_d/tru_delta)
    model_err_r = np.abs(model_err_r/tru_rho)
    
    model_err_d = model_err_d[5:26,:]
    model_err_r = model_err_r[5:26,:]
    
    return model_err_d, model_err_r
        
    # '''graph it'''
    
    # plt.imshow(model_err, cmap='magma')
    # plt.title('Error Location Distribution')
    # plt.ylabel('S from 90% to 110%')
    # plt.xlabel('-.00005 < r < .00005')
    # plt.colorbar()
    # plt.show()
    
    # sns.kdeplot(model_err.reshape((-1)), shade=True)
    # plt.title('Error Size Distribution')
    # plt.ylabel('Error Density')
    # plt.xlabel('Error Size')
    
    # plt.show()
    
    

    












