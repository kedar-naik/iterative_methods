# -*- coding: utf-8 -*-
"""
threshold magnitudes for varying HB operator sizes

Created on Thu May  4 17:16:34 2017

@author: Kedar
"""

from matplotlib import pyplot as plt
import numpy as np
plt.close('all')

# plot where the standard threshold is less than the recipes one
K = np.arange(16)
K_fine = np.linspace(0.0, 15.0)
N_equal_tresholds = (1.0/8.0)*(1.0+np.sqrt(32.0*K_fine + 17.0)+1)
plt.figure()
plt.plot(K_fine, N_equal_tresholds, 'b--')
plt.plot(K, K+1.0, 'c.', label='$N=K+1$')
plt.plot(K, 2.0*K+1.0, 'r.', label='$N=2K+1$')
plt.plot(K, 3.0*K+1.0, 'g.', label='$N=3K+1$')
plt.text(12.5, 4.0, '$\\tau_S>\\tau_R$')
plt.text(12.5, 1.0, '$\\tau_S<\\tau_R$')
plt.xlabel('$K$', fontsize=14)
plt.ylabel('$N$', fontsize=14)
plt.xlim(0.0, 15.0)
plt.legend(loc='best')

# see when the golub threshold is less than the recipes one
s_1 = np.linspace(0.0, 5.0)     # the largest singular value of G
# for the given number of time instances, find the line along which the two 
# thresholds are equal
K_3Kp1 = (1.0/32.0)*(5.0*s_1**2.0 + np.sqrt((s_1**2.0)*(25.0*s_1**2.0 + 32.0)) + 16.0)
K_2Kp1 = (1.0/8.0)*(s_1**2.0 + np.sqrt((s_1**2.0)*(s_1**2.0 + 4.0)) - 4.0)
K_Kp1 = (1.0/32.0)*(3.0*s_1**2.0 + np.sqrt(3.0)*np.sqrt((s_1**2.0)*(3.0*s_1**2.0 + 32.0)) - 16.0)
plt.figure()
plt.plot(s_1, K_3Kp1, 'g', label='$N=3K+1$')
plt.plot(s_1, K_2Kp1, 'r', label='$N=2K+1$')
plt.plot(s_1, K_Kp1, 'c', label='$N=K+1$')
plt.xlabel('$\\sigma_1$', fontsize=14)
plt.ylabel('$K$', fontsize=14)



# see how the greatest singular value varies with the largest
from HB_practice import HB_expansion_matrix, parse_time_discretization_string
np.set_printoptions(precision=3)

omegas = [1.0, 2.1, 3.6]
omegas = [.00200,8985]

# number of angular frequencies being used
K = len(omegas)

# select the time discretization to use for specifying the of time instances
time_discretization = 'use_Nyquist'
time_discretization = 'use_Nyquist-random'
#time_discretization = 'use_Nyquist-'+str(K+1)
time_discretization = 'use_Nyquist-random-'+str(K+1)
time_discretization = 'use_Nyquist-'+str(3*K+1)
time_discretization = 'use_Nyquist-random-'+str(3*K+1)
#time_discretization = 'use_T1'
#time_discretization = 'use_T1-random'
#time_discretization = 'use_T1-'+str(K+1)
#time_discretization = 'use_T1-random-'+str(K+1)
#time_discretization = 'use_T1-'+str(3*K+1)
time_discretization = 'use_T1-random-'+str(3*K+1)

# parse the time discretization string
discretization_style, \
nonuniform_spacing, \
N = parse_time_discretization_string(omegas, time_discretization)

# compute the constant 2N_0/sqrt(N+N_0+1). if the matrix 2-norm of G is above
# this value, then the golub treshold is less than the recipes one
N_0 = 2*K+1
two_norm_boundary = 2.0*N_0/np.sqrt(N+N_0+1.0)


# see how the largest singular value varies w.r.t. shifts  in the baseline set 
# of angular frequencies
baseline_set = list(omegas)
shifts = np.linspace(-0.99*baseline_set[0], 100*baseline_set[0], 200)

average_1_norms = []
average_2_norms = []
average_inf_norms = []
# set the number of samples to evaluate
if nonuniform_spacing:
    N_trials = 50
else:
    N_trials = 1

for shift in shifts:
    # shift the baseline frequency set
    omegas_to_test = [shift + omega_i for omega_i in baseline_set]
    matrix_1_norms = []
    matrix_2_norms = []
    matrix_inf_norms = []
    for i in range(N_trials):        
        # construct the HB expansion matrix, using the desired time discretization
        G = HB_expansion_matrix(omegas_to_test, time_discretization)
        # compute the 1-norm
        G_1_norm = np.linalg.norm(G, ord=1)
        matrix_1_norms.append(G_1_norm)
        # compute the 2-norm
        G_2_norm = np.linalg.norm(G, ord=2)
        matrix_2_norms.append(G_2_norm)
        # compute the infinity norm
        G_inf_norm = np.linalg.norm(G, ord=np.inf)
        matrix_inf_norms.append(G_inf_norm)
    average_1_norms.append(sum(matrix_1_norms)/N_trials)
    average_2_norms.append(sum(matrix_2_norms)/N_trials)
    average_inf_norms.append(sum(matrix_inf_norms)/N_trials)
    
plt.figure()
if N==N_0:
    plt.plot(shifts, average_1_norms, 'g-', label='$\\alpha=1$')
else:
    plt.plot(shifts, average_1_norms, 'g--', label='$\\alpha=1$')
plt.plot(shifts, average_2_norms, 'b--', label='$\\alpha=2$')
plt.plot(shifts, average_inf_norms, 'r--', label='$\\alpha=\\infty$')
plt.plot(shifts, two_norm_boundary*np.ones(len(shifts)), 'm-', label='$2N/\sqrt{N+N_o+1}$')
plt.xlim(-baseline_set[0], shifts[-1])
plt.ylim(0,max(G.shape)+1)
plt.xlabel('$\\Delta \mathbb{\\omega}$', fontsize=14)
if 'random' in time_discretization:
    plt.ylabel('$\overline{||G||_{\\alpha}} \quad (' +str(N_trials)+'\, samples)$', fontsize=14)
else:
    plt.ylabel('$\|G\|_{\\alpha}$', fontsize=14)
title_string = '$\mathbb{\\omega} = {'+str(baseline_set)+' + \mathbb{\\Delta \\omega}}\quad\quad$'

if discretization_style == 'use_T1':
    title_string += '$\\Delta t = \\frac{2\\pi}{\\omega_1 N}$'
else:
    title_string += '$\\Delta t = \\frac{\\omega_K}{\\pi}$'
if nonuniform_spacing:
    title_string += '$\quad arbitrary$'
else:
    title_string += '$\quad uniform$'
N_to_print = str(int((N-1)/K))
if N_to_print == '1': 
    N_to_print = ''
title_string += '$\quad N='+N_to_print+'K+1$'
plt.title(title_string, y=1.01)
plt.legend(loc='best')



# plot the constant marking the 2-norm boundary described above
K = np.arange(5)
N_o = 2.0*K+1.0
N_Kp1 = K + 1.0 
boundary_Kp1 = 2.0*N_o/np.sqrt(N_Kp1+N_o+1) 
boundary_2Kp1 = 2.0*N_o/np.sqrt(N_o+N_o+1)
N_3Kp1 = 3.0*K + 1.0 
boundary_3Kp1 = 2.0*N_o/np.sqrt(N_3Kp1+N_o+1)

plt.figure()
plt.plot(K, N_o, 'k--', label='$N_o=\|G\|_\infty$')
plt.plot(K, N_3Kp1, 'k.-')
plt.plot(K, boundary_Kp1, 'c-', label='$N=K+1$')
plt.plot(K, boundary_2Kp1, 'r-', label='$N=2K+1$')
plt.plot(K, boundary_3Kp1, 'g-', label='$N=3K+1$')
plt.xlabel('$K$', fontsize=14)
plt.ylabel('$2 N_o / \sqrt{N+N_o+1}$', fontsize=14)
plt.legend(loc='best')
