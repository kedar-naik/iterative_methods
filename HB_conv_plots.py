# -*- coding: utf-8 -*-
"""
Created on Sun May 15 04:34:36 2016

@author: Kedar
"""
from matplotlib import pyplot as plt

#-----------------------------------------------------------------------------#
# trends pseudo-timestep size and iterations to convergence for HB

conv_crit = 1e-5            # convergence criteria
omegas = [1.3, 2.5]         # given angularfreqs

# data:   delta_t   n_iterations
runs = [(0.001,   17199),
        (0.002,   8612),
        (0.003,   5751),
        (0.004,   4320),
        (0.005,   3462),
        (0.006,   2890),
        (0.007,   2482),
        (0.008,   2176),
        (0.009,   1938),
        (0.010,   1748),
        (0.011,   1592),
        (0.012,   1463),
        (0.013,   1354),
        (0.014,   1260),
        (0.015,   1179),
        (0.016,   1107),
        (0.017,   1045),
        (0.018,   989),
        (0.019,   939),
        (0.020,   894),
        (0.021,   854),
        (0.022,   817),
        (0.023,   783),
        (0.024,   752),
        (0.025,   724),
        (0.050,   394),
        (0.075,   287),
        (0.100,   240),
        (0.125,   218),
        (0.130,   216),
        (0.135,   213),
        (0.136,   214),
        (0.137,   214),
        (0.138,   213),
        (0.139,   213),
        (0.140,   212),
        (0.141,   212),
        (0.142,   212),
        (0.143,   212),
        (0.144,   212),
        (0.145,   212),
        (0.146,   211),
        (0.147,   211),
        (0.148,   211),
        (0.149,   211),
        (0.150,   212),
        (0.151,   211),
        (0.152,   211),
        (0.153,   211),
        (0.154,   211),
        (0.155,   212),
        (0.156,   212),
        (0.157,   212),
        (0.158,   211),
        (0.159,   212),
        (0.160,   212),
        (0.161,   213),
        (0.165,   214),
        (0.170,   216),
        (0.175,   218),
        (0.200,   240),
        (0.225,   297),
        (0.230,   328),
        (0.235,   344)
        ]
y_label = r'$iterations$'

# parse data
delta_taus = [delta_tau for delta_tau,i in runs]
iters = [iteration for del_tau,iteration in runs]

min_runs = [run for run in runs if run[1]==min(iters)]
min_delta_taus = [delta_tau for delta_tau,i in min_runs]
min_iters = [iteration for del_tau,iteration in min_runs]

# plot data
plot_me = True
if plot_me:
    plot_name = 'iterations_v_delta_tau (conv='+str(conv_crit)+').png'
    plt.figure()
    plt.semilogy(delta_taus,iters,'k.-')
    plt.semilogy(delta_taus[-1],iters[-1],'r*')
    plt.semilogy(min_delta_taus,min_iters,'g.')
    plt.xlabel(r'$\Delta \tau$', fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    title = ''
    counter=1
    for omega in omegas:
        title = title + '$\omega_{'+str(counter)+'} ='+str(omega)+'\quad $'
        counter += 1
    plt.title(title+r'$\quad converged \,\,to \colon \,'+str(conv_crit)+'$')
    print ('saving figure...')
    plt.savefig(plot_name, dpi=500)
    print ('figure saved: ' + plot_name)
    plt.close()
#-----------------------------------------------------------------------------#    
    