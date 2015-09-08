# -*- coding: utf-8 -*-
"""
Created on Fri Sep 04 18:01:09 2015

@author: Kedar
"""
# see what happens to the condition number of the time-spectral operator 
# matrix as the number of time instances, N, and the period of oscillation, T,
# are varied

import numpy as np
from matplotlib import pylab as plt
import math
from time_spectral import time_spectral_operator

plt.close('all')
plt.rc('text', usetex=True)               # for using latex
plt.rc('font', family='serif')            # setting font

# This plot produces non-determininstic results!!!
# The TS operator matrix is so sensitive that rounding errors at the machine
# level are causing large fluctuations in the condition number
runs = 4    # the number of runs (and plots) you'd like to see

# parameters to sweep through
Ts = [0.1, 1.0, 2.0, 2.0*math.pi, 4.0*math.pi]    # period of oscillation
Ns = range(2,41)                     # number of time instances

for run in range(runs):
    plt.figure()
    print 'run #', run+1
    plot_name = 'TS_cond_num - run ' + str(run+1)
    for T in Ts:
        conds = []
        print '\tT = ', T
        for N in Ns:
            # comptue the time-spectral operator for these parameters
            D_t = np.matrix(time_spectral_operator(N,T))
            # find the condition number of this matrix
            conds.append(np.linalg.cond(D_t))
            print '\t\tN = ', N, '; cond(D) = ', conds[-1]
        # plot the results (with the appropriate label)
        if (T/math.pi)%1==0:
            plt.plot(Ns,conds,'.-', label = '$T = '+str(int(T/math.pi))+'\pi$')
        else:
            plt.plot(Ns,conds,'.-', label = '$T = '+str(T)+r'$')
    # use the axes of the first plot for all the others (not elegant, but easy)
    if run == 0:
        x_start, x_end, y_start, y_end = plt.axis()
    plt.axis((x_start, x_end, y_start, y_end))
    plt.xlabel('$N$')
    plt.ylabel('$cond(\mathbf{D_t})$')
    plt.legend(loc='best')
    print 'saving final image...'
    plt.savefig(plot_name, dpi=500)
    print 'figure saved: ' + plot_name