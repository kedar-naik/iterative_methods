# -*- coding: utf-8 -*-
"""
This script allows one to model a periodic function using both the time-
spectral method and the harmonic-balance method.

one thing shown clearly by this script is that using the Nyquist-based time 
discretization is necessary even if you know the exact values of the 
frequencies are known exactly.

Created on Mon Aug 14 21:52:37 2017

@author: Kedar
"""
import numpy as np
from matplotlib import pyplot as plt
plt.ioff()
import webbrowser
from time_spectral import fourierInterp
from dft_practice import my_dft
from HB_practice import harmonic_balance_operator, trig_interp_given_freqs, HB_omega_check

#-----------------------------------------------------------------------------#
def TS_from_HB_operator(K, N, T):
    '''
    build the time-spectral operator when N != 2K+1
    (also works for the standard, N=2K+1)
    '''
    # determine the locations of the N, uniformly spaced time instances that 
    # just span T
    t = np.arange(N)*T/N
    # compute the differentiation matrix
    N_o = 2.0*K+1.0
    D = np.zeros((N,N))
    for l in range(N):
        for j in range(N):
            if l==j:
                D[l][j] = 0.0
            else:
                chi = 2.0*np.pi*(l-j)/N
                D[l][j] = (2.0*np.pi/(T*N))*(np.sin(chi/2.0)*(N_o/2.0)*np.cos(N_o*chi/2.0)-np.sin(N_o*chi/2.0)*(1.0/2.0)*np.cos(chi/2.0))/(np.sin(chi/2.0)**2.0)
    # return the operator matrix and the time instances
    return D, t
#-----------------------------------------------------------------------------#
def my_harmonic_periodic_fun(t, T, n_harmonics=1):
    '''
    this function returns a periodic function at the supplied time points. but,
    it allows you to also specify the number of harmonics of the fundamental
    frequency (2*pi/T) to include. the fundamental frequency corresponds to the
    user-defined period. note the nomenclature! to only use the fundamental 
    frequency, set n_harmonics=1. to include higher harmonics, increase the
    n_harmonics argument.
    function being sampled: 
          f(t) = cos(k*2*pi*t/T) + sin(k*2*pi*t/T) + higher harmonics
    Inputs:
        - t:            time instances where samples are needed
        - T:            desired period of oscillation
        - n_harmonics:  desired number of harmonics to include
    Output:
        - f:    periodic function with the desired number of harmonics
        - df:   derivative of the periodic function
    '''
    # compute the fundamental frequency
    omega_fundamental = 2.0*np.pi/T
    # build up the function and its analyical derivative, term-by-term
    f = 0.0
    df = 0.0
    power = 3
    for k in range(1, n_harmonics+1):
        f += np.sin(k*omega_fundamental*t)**power + np.cos(k*omega_fundamental*t)**power
        df += power*k*omega_fundamental*(np.sin(k*omega_fundamental*t)**(power-1)*np.cos(k*omega_fundamental*t) - np.cos(k*omega_fundamental*t)**(power-1)*np.sin(k*omega_fundamental*t))
    # latex string of the equation
    equation_string = '$f(t)=\sum_{k=1}^{'+str(n_harmonics)+'}\\left[\sin^{'+str(power)+'}\\left(k\\frac{2\pi}{T}t\\right) + \cos^{'+str(power)+'}\\left(k\\frac{2\pi}{T}t\\right)\\right]$'
    # return the function, the derivative, and the latex string
    return f, df, equation_string
#-----------------------------------------------------------------------------#

# set the different constants to test
T = 17.0        # period of oscillation
K = 1           # number of HB frequencies
N_o = 2*K+1     # N_0 = 2K+1
N = 3*K+1       # if you want to "oversample," i.e. have N > N_0 = 2K+1
#N = N_o        # if you want N = N_0, uncomment this line

# initialize the HB expansion matrix, G
G = np.zeros((N,N_o), dtype=np.complex_)
# fill in the matrix, element-by-element
for i in range(N):
    for j in range(-K,K+1):
        G[i][j] = np.exp(2.0*np.pi*1j*j*i/N)

# build the operator and get the time instances
D_ts, t_ts = TS_from_HB_operator(K, N, T)

# define the ACTUAL signal's constants
K_input = 3
T_actual = 17.0     # [s]
# sample the actual signal at the time instances
f_actual_ts, dfdt_actual_ts, eq_str = my_harmonic_periodic_fun(t_ts, T_actual, n_harmonics=K_input)

# define the fine time grid
n_fine = 700
t_fine = np.linspace(0, T_actual, n_fine)
# sample the actual signal on the fine time grid
f_actual_fine, dfdt_actual_fine, eq_str = my_harmonic_periodic_fun(t_fine, T_actual, n_harmonics=K_input)

# use the operator to compute the derivative at the time instances
dfdt_ts = np.dot(D_ts,f_actual_ts)

# build the HB operator using the actual frequencies.
# to find the actual frequencies, very finely sample three periods of the 
# actual signal and then find the peaks. first, define the fine time grid:
t_fine_per = np.linspace(0, 3.0*T_actual, 2**10)
# sample the actual signal over this long, fine time grid
f_actual_fine_per, dfdt_actual_fine_per, eq_str = my_harmonic_periodic_fun(t_fine_per, T_actual, n_harmonics=K_input)

# thesis plots: DFT of three periods of the exact signal and plot the specturm
s, F, powers, peaks_found, peak_boundaries = my_dft(t_fine_per, f_actual_fine_per, 
                                                 percent_energy_AC_peaks=97.0,
                                                 shift_frequencies=True,
                                                 use_angular_frequencies=True,
                                                 plot_spectrum=True, 
                                                 plot_log_scale=False,
                                                 refine_peaks=False,
                                                 auto_open_plot=True,
                                                 verbose=True,
                                                 title_suffix='',
                                                 plot_suffix=' - dfdt',
                                                 use_fft=True,
                                                 new_x_lims=[-4,4],
                                                 plot_together=False)

# determine the frequencies by rounding the peaks found. should get:
# omegas = [0.3696,  0.7392,  1.1088, 2.2176, 3.3264] # [rad/s] (exact values)
# omegas = [0.369,   0.738,   1.108,  2.215,  3.323] # [rad/s], (expected 
# values from spectrum (np.round(peaks_found,3)))
omegas = list(np.round(peaks_found, 3))

# check to make sure these frequencies are not inadmissible!
HB_omega_check(omegas)

# define how to select the harmonic-balance time instances
my_time_discretization = 'use_T1'
my_time_discretization = 'use_Nyquist'

# set the number of HB time instances to use
N_hb = 2*len(omegas)+1

# create the HB operator and define the time instances
D_hb, t_hb = harmonic_balance_operator(omegas, time_discretization=my_time_discretization)

# sample the actual function at the HB time instances
f_actual_hb, dfdt_actual_hb, eq_str = my_harmonic_periodic_fun(t_hb, T_actual, n_harmonics=K_input)

# apply the HB operator to compute the discrete derivative
dfdt_hb = np.dot(D_hb,f_actual_hb)

# plot the results
plot_name = 'TS-v-HB_dashboard'
fig = plt.figure(plot_name)
width, height = fig.get_size_inches()
fig.set_size_inches(1.5*width, 1.5*height, forward=True)
fig.suptitle(eq_str, fontsize=16, y=0.57)
fig.subplots_adjust(hspace=1.0, wspace=0.3)
# exact function with TS samples
plt.subplot(2,2,1)
plt.plot(t_ts,f_actual_ts,'ko',label='$f_{TS_i}$')
plt.plot(t_fine, f_actual_fine, 'k-', label='$f$')
plt.xlabel('$t$, [$s$]', fontsize=14)
plt.ylabel('$f$', fontsize=14)
plt.legend(loc='best')
plt.title('$T_{actual}='+str(T_actual)+'$')
# exact derivative
plt.subplot(2,2,2)
plt.plot(t_fine, dfdt_actual_fine, 'b-', label='$df/dt$')
# TS derivative
plt.plot(t_ts, dfdt_ts, 'ro', label='$df/dt_{TS_i}$')
t_fine_ts = np.linspace(0.0,T,n_fine) # in case your TS period is wrong
plt.plot(t_fine_ts, fourierInterp(t_ts,dfdt_ts,t_fine_ts)[1], 'r--', label='$fourier \; interp.$')
plt.xlabel('$t$, [$s$]', fontsize=14)
plt.ylabel('$\\frac{df}{dt}$', fontsize=14)
plt.legend(loc='best')
if N==2*K+1:
    plt.title('$T='+str(T)+'\,s\;,\;K='+str(K)+'\;,\;N='+str(N)+'\;,\;N=N_o\quadD_{TS}\in\mathbb{R}^{'+str(N)+'\\times'+str(N)+'}$')
else:
    plt.title('$T='+str(T)+'\,s\;,\;K='+str(K)+'\;,\;N='+str(N)+'\quadD_{TS}\in\mathbb{R}^{'+str(N)+'\\times'+str(N)+'}$')
# exact function with HB samples
plt.subplot(2,2,3)
plt.plot(t_hb,f_actual_hb,'ko', label='$f_{HB_i}$')
plt.plot(t_fine,f_actual_fine,'k-', label='$f$')
plt.xlabel('$t$, [$s$]', fontsize=14)
plt.ylabel('$f$', fontsize=14)
plt.legend(loc='best')
if my_time_discretization=='use_T1':
    plt.title('$uniform \, spacing \, over \, T_1$')
if my_time_discretization=='use_Nyquist':
    plt.title('$uniform \; spacing \; satisfying \; Nyquist$')
# exact derivative
plt.subplot(2,2,4)
plt.plot(t_fine, dfdt_actual_fine, 'b-', label='$df/dt$')
# HB derivative
plt.plot(t_hb, dfdt_hb, 'go', label='$df/dt_{HB_i}$')
t_fine_hb = np.linspace(0.0, t_hb[-1], n_fine)
plt.plot(t_fine_hb, trig_interp_given_freqs(t_hb,dfdt_hb, omegas, x_int=t_fine_hb)[1], 'g--', label='$trig. \; interp.$')
plt.xlabel('$t$, [$s$]', fontsize=14)
plt.ylabel('$\\frac{df}{dt}$', fontsize=14)
plt.legend(loc='best')
plt.title('$using \; \omega \; values \; from \; DFT$')
# save figure
plt.savefig(plot_name+'.png', dpi=400)
plt.close(plot_name)
webbrowser.open(plot_name+'.png')

# ------ THESIS PLOTS (different colors from dashboard) ------

# set the font size for the legend and axis labels
the_fontsize = 18

# thesis plots: periodicity plot
t_fine_per = np.linspace(0,2.5*T_actual,n_fine)
f_actual_fine_per, dfdt_actual_fine_per, eq_str = my_harmonic_periodic_fun(t_fine_per, T_actual, n_harmonics=K_input)
plot_name = 'exact_function_periods'
plt.figure()
plt.plot(t_fine_per, f_actual_fine_per, 'k-', label='$f$')
plt.xlabel('$t, \\left[\,\\mathrm{s}\,\\right]$',fontsize=the_fontsize)
plt.ylabel('$f$', fontsize=the_fontsize)
plt.ylim(-3,4)
plt.savefig(plot_name+'.png', dpi=400)
plt.close()
webbrowser.open(plot_name+'.png')

# thesis plots: TS time instances
plot_name = 'demo_ts_samples_'+str(N)
plt.figure()
plt.plot(t_fine, f_actual_fine, 'k-', label='$f$')
plt.plot(t_ts,f_actual_ts,'bo',label='$f_{TS_i}$')
plt.xlabel('$t, \\left[\,\\mathrm{s}\,\\right]$',fontsize=the_fontsize)
plt.ylabel('$f(t)$', fontsize=the_fontsize)
plt.ylim(-3,4)
plt.legend(loc='best', fontsize=the_fontsize)
plt.savefig(plot_name+'.png', dpi=400)
plt.close()
webbrowser.open(plot_name+'.png')

# thesis plots: TS derivative
plot_name = 'demo_ts_derivative_'+str(N)
plt.figure()
plt.plot(t_fine, dfdt_actual_fine, 'r-', label='$df/dt$')
plt.plot(t_ts, dfdt_ts, 'bo', label='$df/dt_{TS_i}$')
t_fine_ts = np.linspace(0.0,T,n_fine) # in case your TS period is wrong
plt.plot(t_fine_ts, fourierInterp(t_ts,dfdt_ts,t_fine_ts)[1], 'b--', label='$fourier \; interp.$')
plt.xlabel('$t, \\left[\,\\mathrm{s}\,\\right]$', fontsize=the_fontsize)
plt.ylabel('$\\frac{df}{dt}$', fontsize=the_fontsize)
plt.legend(loc='best', fontsize=the_fontsize)
plt.savefig(plot_name+'.png', dpi=400)
plt.close()
webbrowser.open(plot_name+'.png')

# thesis plots: exact function with HB samples
if my_time_discretization=='use_Nyquist':
    plot_name = 'demo_hb_samples_Nyquist'
if my_time_discretization=='use_T1':
    plot_name = 'demo_hb_samples_T1'
plt.figure()
plt.plot(t_fine,f_actual_fine,'k-', label='$f$')
plt.plot(t_hb,f_actual_hb,'go', label='$f_{HB_i}$')
plt.xlabel('$t, \\left[\,\\mathrm{s}\,\\right]$', fontsize=the_fontsize)
plt.ylabel('$f$', fontsize=the_fontsize)
plt.ylim(-3,4)
plt.legend(loc='best', fontsize=the_fontsize)
plt.savefig(plot_name+'.png', dpi=400)
plt.close()
webbrowser.open(plot_name+'.png')

# thesis plots: HB derivative
if my_time_discretization=='use_Nyquist':
    plot_name = 'demo_hb_derivative_Nyquist'
if my_time_discretization=='use_T1':
    plot_name = 'demo_hb_derivative_T1'
plt.figure()
plt.plot(t_fine, dfdt_actual_fine, 'r-', label='$df/dt$')
plt.plot(t_hb, dfdt_hb, 'go', label='$df/dt_{HB_i}$')
t_fine_hb = np.linspace(0.0, t_hb[-1], n_fine)
t_fine_hb = t_fine
plt.plot(t_fine_hb, trig_interp_given_freqs(t_hb,dfdt_hb, omegas, x_int=t_fine_hb)[1], 'g--', label='$trig. \; interp.$')
plt.xlabel('$t, \\left[\,\\mathrm{s}\,\\right]$', fontsize=the_fontsize)
plt.ylabel('$\\frac{df}{dt}$', fontsize=the_fontsize)
plt.legend(loc='best', fontsize=the_fontsize)
plt.savefig(plot_name+'.png', dpi=400)
plt.close()
webbrowser.open(plot_name+'.png')