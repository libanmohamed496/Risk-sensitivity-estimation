# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 12:41:53 2020

@author: liban
"""

import random
import numpy as np
from gmab_model import  read_shock_data
import matplotlib.pyplot as plt
import gmab_model
from gmab_model import set_world
from gmab_model import vec_rollforward, vec_GMAB, vec_GMAB_ratchet
from gmab_model import delta, rho
from gmab_model import value_by_rep, value_by_error, value_by_precision
from gmab_model import shock_value, write_shock_data, read_shock_data

def averages(file, N, DelRho, args):
    world = read_shock_data(file)[0] + (None,)
    shock_dat = read_shock_data(file)[1]
    range_r=21                                          # number of r shocks
    range_s=31                                          # number of s shocks
    SR=np.array([[shock_dat[r+s*range_r][0] for r in range(range_r)] for s in range(range_s)])
    # return (SR)
    V=np.array([[shock_dat[r+s*range_r][2][0] for r in range(range_r)] for s in range(range_s)])
    # return (V)
    a0, b0, a1, b1, a2, b2, a3, b3, a4, b4 = 2, 2, 4, 20, 16, 11, 28, 0, 28, 16
    t_max = 10 # t in years
    dt = 1/252
    
    shocks = [tuple(SR[a0, b0]),tuple(SR[a1, b1]),tuple(SR[a2, b2]),tuple(SR[a3, b3]),tuple(SR[a4, b4])]
    raw = shock_value(world, shocks, value_by_rep, N ,DelRho, .001, vec_GMAB_ratchet, *args)
    B = np.zeros(5)
    for i in range(5):
        B[i] = raw[i][2][0]
        MCNoise = (abs(V[a0,b0]-B[0]) + abs(V[a1,b1]-B[1]) + abs(V[a2,b2]-B[2])+ abs(V[a3,b3]-B[3])+ abs(V[a4,b4]-B[4]))/5
    P = np.zeros((5,2), dtype = int)
    P[0] = a0, b0
    P[1] = a1, b1
    P[2] = a2, b2
    P[3] = a3, b3
    P[4] = a4, b4
    srp = np.zeros((5,2), dtype = float)
    for i in range(5):
        for j in range(2):
            srp[i,j] = SR[P[i,0],P[i,1],j]
            Q = (srp-srp[0])[1:]
    W = np.zeros((4,4),dtype=float)
    for i in range(4):
        W[i] = Q[i,0], Q[i,1], Q[i,0]**2, Q[i,0]*Q[i,1]
        s0 , r0 = SR[P[0,0],P[0,1]]
    v0 = B[0]
    C = B[1:]- v0
    L = np.linalg.solve(W, C)
    def estimate(sx,rx):
        return v0 + L[0] * (sx-s0) + L[1] * (rx-r0) + L[2] * (sx-s0)**2 + L[3] * (sx-s0) * (rx-r0)
    predicts = np.array([[estimate(SR[s,r,0],SR[s,r,1]) for r in range(range_r)]for s in range(range_s)])
    errors = V - predicts
    errors = np.abs(errors)/V
    # print(np.mean(V))
    # MSE = np.sum(np.square(V- predicts)) / (range_s*range_r)
    return errors[5:26]

av_mse = 0
av_noise = 0
# for i in range(100):
#     av = averages('GMAB_ratchet_samples/sample2_rho_.csv',5,rho, args)
#     av_mse += av[1]
#     av_noise += av[0]

av_noise /= 100
av_mse /= 100
print(av_mse)
print(av_noise)
args = 0,10,np.ones(int(10*252)+1),.045,0,.01

def five_pt_reg(v_file, del_file, rho_file, N, args):
    return averages(del_file,int(N/15),delta, args), averages(rho_file,int(N/15),rho, args)
    
    
    
    
    
    