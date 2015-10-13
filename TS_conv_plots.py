# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 18:15:35 2015

@author: Kedar
"""

from matplotlib import pylab as plt

plt.close('all')
plt.rc('text', usetex=True)               # for using latex
plt.rc('font', family='serif')            # setting font

# for the 1D ODE, plot residual stall level against given (incorrect) period

# periods tried (2.0 is correct)
T = [1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]

# residual at which analysis stalls
stall_res = [3.00822747148e-11,    # T = 1.0
             3.40277508912e-11,    # T = 1.5
             3.45649060659e-11,    # T = 1.75
             3.47605031684e-11,    # T = 2.0
             3.51394994588e-11,    # T = 2.25
             3.54063857616e-11,    # T = 2.5
             3.55316119477e-11]    # T = 3.0
             
# iterations required to reach that stalled residual
iterations = [721331,    # T = 1.0
              693673,    # T = 1.5
              691735,    # T = 1.75
              684037,    # T = 2.0
              686417,    # T = 2.25
              688222,    # T = 2.5
              690531]    # T = 3.0

# plot the stalled residual values vs. the period
plot_name = 'stallRes_v_T'
plt.figure()
plt.plot(T, stall_res, 'k.-')
plt.plot(2.0,3.47605031684e-11,'r*')
plt.xlabel(r'$T$', fontsize=18)
plt.ylabel(r'$\|R\|_{stall}$', fontsize=18)
print 'saving figure...'
plt.savefig(plot_name, dpi=500)
print 'figure saved: ' + plot_name
plt.close()

# plot the iterations required to reach stalled residual vs. the period
plot_name = 'iterations_v_T'
plt.figure()
plt.plot(T, iterations, 'k.-')
plt.plot(2.0,684037,'r*')
plt.xlabel(r'$T$', fontsize=18)
plt.ylabel(r'$iterations \,\,\, required$', fontsize=18)
print 'saving figure...'
plt.savefig(plot_name, dpi=500)
print 'figure saved: ' + plot_name
plt.close()