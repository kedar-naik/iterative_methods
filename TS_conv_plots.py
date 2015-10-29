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

###############################################################################
# period finding via partial convergence ######################################
###############################################################################
delta_tau = 0.0001            # pseudo-timestep used 
conv_crit = 1e1               # partial convergence level
T_actual = 2.0                # correct T
# data:   T     iterations
runs = [(0.25,   25462),
        (0.50,	25401),
        (0.75,	25376),
        (1.00,	25400),
        #(1.50,	25334),
        #(1.75,	25328),
        (2.00,	25302),
        #(2.25,	25314),
        #(2.50,	25320),
        #(2.75,	25318),
        (3.00,	25331),
        #(3.25,	25323),
        #(3.50,	25314),
        #(3.75,	25312),
        (4.00,	25302),
        #(4.25,	25302),
        #(4.50,	25309),
        #(4.75,	25309),
        (5.00,	25313),
        #(5.25,	25315),
        #(5.50,	25305),
        #(5.75,	25303),
        (6.00,	25302),	
        #(6.25,	25289),
        #(6.50,	25295),
        #(6.75,	25287),
        (7.00,	25337),
        #(7.25,  25314),
        #(7.50,	25308),
        #(7.75,	25308),
        (8.00,	25302),
        (9.00,	25298),
        (10.00,	25303),
        (11.00,	25214),
        (12.00,	25303),
        (13.00,	25312),
        (14.00,	25303),
        (15.00,	25313),
        (16.00,	25302),
        (17.00,	25302),
        (18.00,	25302),
        (19.00,	25293),
        (20.00,	25303),
        (21.00,	25294),
        (22.00,	25303),
        (23.00,	25393),
        (24.00,	25303),
        (25.00,	25308),
        (26.00,	25303),
        (48.00,  25303),
        (49.00,	25313),
        (50.00,	25303),
        (51.00,	25302),
        (74.00,	25304),
        (75.00,	25338),
        (76.00,	25304),
        (100.00,	25304)]
        
        
#-----------------------------------------------------------------------------        
 
'''
delta_tau = 0.00005            # pseudo-timestep used 
conv_crit = 1e1               # partial convergence level
T_actual = 5.25                # correct T
# data:   T     iterations
runs = [(1.00,	50865),
        (2.00,	50755),
        (3.00,	50756),
        (4.00,	50669),
        (5.00,	50627),
        (5.25,	50607),
        (6.00,	50636),
        (7.00,	50640),
        (8.00,   50668),
        (9.00,	50631),
        (10.00,	50626),
        (11.00,	50600),
        (12.00,	50620),
        (13.00,	50623),
        (14.00,	50620),
        (15.00,	50609),
        (16.00,	50588),
        (17.00,	50601),
        (18.00,	50646)]
'''



Ts = [period for period,i in runs]
iters = [i for period,i in runs]
iter_actual = [i for period,i in runs if period == T_actual][0]

plot_name = 'iter_v_T (conv='+str(int(conv_crit))+')'
plt.figure()
plt.plot(Ts,iters,'k.-')
plt.plot(Ts,[iter_actual]*len(Ts),'r-')
plt.plot(T_actual,iter_actual,'r*')
plt.xlabel(r'$T$', fontsize=18)
plt.ylabel(r'$iterations \,\,\, required$', fontsize=18)
plt.title(r'$converged \,\,to \colon \,\,$'+str(conv_crit)+r'$; \,\, \Delta \tau = \,$'+str(delta_tau)+r'$; \,\, T_{actual}= \,$'+str(T_actual))
print 'saving figure...'
plt.savefig(plot_name, dpi=500)
print 'figure saved: ' + plot_name
plt.close()