# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 20:26:39 2020

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

from trading_grid import TGrid
from cubic_spline import cube_spline
from five_pt_reg import five_pt_reg
from gaussian_process_reg import GPR


v_file = 'GMAB_ratchet_samples/sample5_.csv'
del_file = 'GMAB_ratchet_samples/sample5_delta_.csv'
rho_file = 'GMAB_ratchet_samples/sample5_rho_.csv'
args = 0, 10, np.ones(2520*10), .045, 0, .01

methods = [TGrid, cube_spline, five_pt_reg, GPR]
names = ['Trading Grid', 'Cubic Spline', 'Quadratic Interpolation', 'Regression Kriging']

algo_tups = list(zip(it.count(),methods))

MC_tups = [(j,100*2**j) for j in range(11)]

# model, comp, iters, del/rho, S, r
# (4, 11, 20, 2, 21, 21)

# results = np.zeros((4, 11, 20, 2, 21, 21))

# for algo in [(0,TGrid),(1,cube_spline)]:
#     for MC in MC_tups:
#         for j in range(20):
#             print(algo, MC, j)
#             results[algo[0],MC[0],j] = algo[1](v_file, del_file, rho_file, MC[1], args)








# # rel MAE vs comp time
# for j in range(1,4):
#     plt.plot([np.mean(more_results[j,k,]) for k in range(10)])
# plt.legend(names[1:3])
# plt.xlabel('MC simulations')
# xslice = slice(3,10,3)
# plt.xticks(list(range(10))[xslice],comps[xslice])
# plt.ylabel('Absolute Error')



# density plot at different, specific comp times

# for j in range(1,4):
#     sns.kdeplot(np.log(more_results[j,7,:,0]).reshape((-1,)), shade=True, label=names[j])
# plt.xlabel('Log Normalized Mean Absolute Error')
# plt.ylabel('Error Size Density')
# plt.show()


# hmd_l = five_pt_reg(del_file, rho_file, 10000, args)[0]
# hmd_g = GPR(del_file, rho_file, 10000, args, num_points=5, diff=-1)[0]
# plt.imshow(hmd_g)
# plt.colorbar()

