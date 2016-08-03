# -*- coding: utf-8 -*-
"""
Created on Mon May 16 14:44:22 2016
@author: Kedar R. Naik
"""

import csv
import numpy as np
from matplotlib import pyplot as plt
plt.close('all')

#-----------------------------------------------------------------------------#
def fourierInterp_given_freqs(x, y, omegas, x_int=False):
    '''
    This function interpolates a given set of ordinates and abscissas with a
    Fourier series that uses a specific set of frequencies. The interpolation 
    is constructed using coefficients found for the cosine and sine terms by
    solving a linear system built up from the given abscissas and ordinates.
    Input:
      - abscissas, x (as a list) (leave out last, duplicate point in period)
      - ordinates, y (as a list) (again, leave out last point, if periodic)
      - angular frequencies, omegas (as a list)
      - new abscissas, x_int (as a list) (optional! defaults to 10x refinement)
    Output:
      - new abscissas, x_int (as a list)
      - interpolated ordinates, y_int (as a list)
      - derivative of the interpolant, dydx_int (as a list)
    '''
    import math
    import numpy as np
    # refinment factor for the interpolant. (If there are originally 10 points
    # but you want the Fourier Series to be defined on 50 points, then the 
    # refinement factor is 5.)
    refine_fac = 10
    # number of points passed in
    n = len(x)                  
    # if zero has not been included as a frequency, add it
    if 0 not in omegas:
        omegas = [0.0] + omegas
    # total number of frequencies being considered, including the D.C. value
    m = len(omegas)
    # compute the coefficients by setting up and solving a linear system
    N = 2*(m-1)+1
    A = np.zeros((N,N))
    b = np.zeros((N,1))
    for i in range(N):
        b[i] = y[i]
        for j in range(N):
            if j == 0:
                A[i][j] = 1
            else:
                if j%2 == 1:
                    A[i][j] = math.cos(omegas[int((j+1)/2)]*x[i])
                else:
                    A[i][j] = math.sin(omegas[int(j/2)]*x[i])
    Fourier_coeffs = np.linalg.solve(A,b)
    print('\n\tCondition # used to find fourier coeffs: cond(A) = ',np.round(np.linalg.cond(A),2))
    # create separate lists for the cosine and sine coefficients
    a = []
    b = [0]
    for i in range(N):
        if i==0 or i%2==1:
            a.append(Fourier_coeffs[i][0])
        else:
            b.append(Fourier_coeffs[i][0])
    # set x_int, if it hasn't been given
    if type(x_int) == bool:
        n_int = refine_fac*(n)
        x_int = np.linspace(x[0],x[-1]+x[1],n_int)
    else:
        n_int = len(x_int)
    # find the actual interpolation
    y_int = [0.0]*n_int
    dydx_int = [0.0]*n_int
    for i in range(n_int):
        y_int[i] = a[0]        # the "DC" value
        dydx_int[i] = 0.0      # intialize the summation
        for j in range(m-1):
            y_int[i] += a[j+1]*math.cos(omegas[j+1]*x_int[i]) + \
                        b[j+1]*math.sin(omegas[j+1]*x_int[i])
            dydx_int[i] += omegas[j+1]* \
                           (b[j+1]*math.cos(omegas[j+1]*x_int[i]) - \
                            a[j+1]*math.sin(omegas[j+1]*x_int[i]))
    return (x_int, y_int, dydx_int)
#-----------------------------------------------------------------------------#
def whittaker_shannon_interp(t,f,t_int=False):
    '''
    this subroutine implements the Whittaker-Shannon interpolation formula.
    Input:
      - abscissas, t (as a list) (leave out last, duplicate point in period)
      - ordinates, f (as a list) (again, leave out last point, if periodic)
      - new abscissas, f_int (as a list) (optional! defaults to 10x refinement)
    Output:
      - new abscissas, t_int (np array)
      - interpolated ordinates, f_int (np array)
    '''
    import numpy as np
    # refinment factor for the interpolant. (If there are originally 10 points
    # but you want the interpolation to be defined on 50 points, then the 
    # refinement factor is 5.)
    refine_fac = 4
    # number of points passed in
    n = len(t)  
    # set t_int, if it hasn't been given
    if type(t_int) == bool:
        n_int = refine_fac*(n)
        t_int = np.linspace(t[0],t[-1],n_int)
    else:
        n_int = len(t_int)
    # implementation of the formula
    delta_t = t[1]
    f_int = []
    for t_i in t_int:
        f_int.append(sum([f_k*np.sinc((1/delta_t)*(t_i-t_k)) for t_k,f_k in zip(t,f)]))
       
    return (t_int, f_int)
#-----------------------------------------------------------------------------#
    
# if you don't have a CSV file full of points, define a signal (use fine grid!)
points = 960
t = np.linspace(0,2*np.pi,points)
omegas_actual = [2.0, 3.3]
f = sum([np.sin(omega*t)+1.5*np.cos(omega*t)**1 for omega in omegas_actual]) + 5
 
# otherwise, provide some details about the CSV file and set flag to True
signal_from_CSV = False
csv_filename = 'f'
analyze_column = 'Close'

# in either case, set some constraints on the reconstruction
energy_cutoff = 1e-1
use_K_frequencies = 5       # not including the DC component

# forcasting options ('N_points', 'whole_signal')
forecast = 'N_points'
#forecast = 'whole_signal'

# foracsting with the integer frequencies filtered from the DFT
use_DFT_freqs = True
# forcasting with set of actual omegas (only if using a constructed function)
use_actual_omegas = True
# interpolation using whittaker-shannon ('whole_signal' is selected by default)
use_whittaker_shannon = False



if signal_from_CSV:
    # read in the results file
    history = []
    with open(csv_filename+'.csv','r') as history_file:
        history_reader = csv.reader(history_file)
        for row in history_reader:
            history.append(row)
    # remove the header row
    header = history.pop(0)
    print('\n\tCSV file read in')
    
    # figure out which columnn contains the desired data
    column_counter = 0
    for entry in header:
        if entry == 'Date':
            date_index = column_counter
        if entry == analyze_column:
            index = column_counter
            break
        column_counter += 1
    
    # extract the date and column to be analyzed from the data
    dates = np.array([row[date_index] for row in history])
    data = np.flipud(np.array([float(row[index]) for row in history]))
    print('\n\tdata extracted')
    
    # define the signal in the time domain
    t = dates
    f = data
    # redefine the number of points
    points = len(f)
    # temporary time vector while i figure out how to 
    t = np.linspace(0,len(f),len(f))
print('\n\tsignal defined')

# take the Fourier transform and compute the energy spectrum
Ff = np.fft.fft(f)
energies = np.absolute(Ff)**2
print('\n\tDFT and energy spectrum computed')

# sets up the mirrored frequency vector
omegas = np.fft.fftfreq(points,1/points)

# filter the spectrum by zeroing out everthing below the cutoff
Ff_filtered = [Ff[0]] # always include the DC component
retained_entry_counter = 1
for i in range(1,len(energies)): 
    if energies[i] > energy_cutoff:
        Ff_filtered.append(Ff[i])
        retained_entry_counter += 1
    else:
        Ff_filtered.append(0.0)
Ff_filtered = np.array(Ff_filtered)
energies_filtered = np.absolute(Ff_filtered)**2
cutoff_active = True

# if the number of active frequencies (aside from the DC component) larger than 
# the maximum frequecies desired to be retained. zero out all but the 
# 2*use_K_frequencies spectrum points with the highest energy
if 2*use_K_frequencies < retained_entry_counter-1:
    cutoff_active = False
    sorted_filtered_energies = np.sort(energies_filtered[1:])
    energies_to_retain = sorted_filtered_energies[-(2*use_K_frequencies):]
    min_retained_energy = energies_to_retain[0]
    for i in range(1,len(energies)):
        if energies[i] < min_retained_energy:
            Ff_filtered[i] = 0.0
    energies_filtered = np.absolute(Ff_filtered)**2

# extract the frequencies that did not get filtered out (clunky...oh well)
omegas_filtered = []
for i in range(points):
    if energies_filtered[i] > 0.0:
        omegas_filtered = np.append(omegas_filtered, omegas[i])
omegas_filtered = [omega for omega in omegas_filtered if omega > 0]

# take the inverse Fourier transform of the filtered Fourier transform
f_recon = np.fft.ifft(Ff_filtered)

# the reconstructed signal is real, but python returns some points with very 
# small complex components. cast those values as real if they're within three 
# orders of magnitude away from machine zero
f_recon = np.real_if_close(f_recon, tol=1000)

# limited forecasting: select N = 2K+1 samples from the IDFT reconstruction at
# a sampling rate just greater than the nyquist rate (2B)
K = len(omegas_filtered)
N = 2*K+1
max_omega = max(omegas_filtered)   # nyquist rate [rad/s]
B = max_omega/(2.0*np.pi)          # nyquist rate [1/s]
nyquist_rate = 2*B
scaling_fac = 3.1     # this needs to be > 1.0 to meet the nyquist criterion
nyquist_delta_t = 1.0/(scaling_fac*nyquist_rate)
if forecast == 'N_points':
    t_nyquist = np.array([t[0]+i*nyquist_delta_t for i in range(N)])
# forecasting based on fitting the entire reconstructed signal
if forecast == 'whole_signal' or use_whittaker_shannon:
    scale_down_fac = 2
    points = int((t[-1]-t[0])/nyquist_delta_t) 
    t_nyquist = np.array([t[0]+i*nyquist_delta_t for i in range(points)])
# linearly interpolate the reconstructed signal onto the time points
f_nyquist = np.interp(t_nyquist,t,f_recon)
# generate a time grid that is 1.5 times longer than the reconstructed signal
t_long = np.arange(t[0],1.5*t[-1],nyquist_delta_t/10.0)
# using the frequencies given, extrapolate from the sample points
if use_DFT_freqs:
    t_long, f_long, dfdt_long = fourierInterp_given_freqs(t_nyquist, f_nyquist, omegas_filtered, t_long)
if use_actual_omegas:
    t_long, f_long, dfdt_long = fourierInterp_given_freqs(t_nyquist, f_nyquist, omegas_actual, t_long)
if use_whittaker_shannon:
    t_long,f_long = whittaker_shannon_interp(t_nyquist,f_nyquist)

# print some information about the reconstruction to the screen
if cutoff_active:
    print('\n\treconstruction limited by min. energy cutoff:',energy_cutoff)
else:
    print('\n\treconstruction limited by number of retained frequencies:', use_K_frequencies)

# plot the signal
plt.figure()
plt.subplot(1,2,1)
plt.plot(t,f,'b-',label='$original \, signal$')
plt.xlabel('$t$', fontsize=16)
plt.ylabel('$f(t)$', fontsize=16)

# plot the energy spectrum
plt.subplot(1,2,2)
plt.semilogy(np.fft.fftshift(omegas),np.fft.fftshift(energies),'bo',label='$power \, spectrum$')
plt.vlines(np.fft.fftshift(omegas),[10**np.floor(np.log10(min(energies)))]*points,np.fft.fftshift(energies),'b')
plt.xlabel('$\omega$', fontsize=16)
plt.ylabel('$|Ff(\omega)|^2$', fontsize=16)

# plot the energy spectrum of the filtered Fourier transform
plt.subplot(1,2,2)
plt.semilogy(np.fft.fftshift(omegas),np.fft.fftshift(energies_filtered),'ro',label='$filtered \, spectrum$')
plt.vlines(np.fft.fftshift(omegas),[10**np.floor(np.log10(min(energies)))]*points,np.fft.fftshift(energies_filtered),'r')
if cutoff_active:
    plt.semilogy(np.fft.fftshift(omegas),[energy_cutoff]*len(energies),'r--')
plt.xlabel('$\omega$', fontsize=16)
plt.ylabel('$|Ff(\omega)|^2$', fontsize=16)
plt.legend()

# plot the reconstructed signal
plt.subplot(1,2,1)
plt.plot(t,f_recon,'r--',label='$reconstruction$')
plt.xlabel('$t$', fontsize=16)
plt.ylabel('$f(t)$', fontsize=16)
plt.tight_layout()

# plot the interpolated/extrapolated signal
plt.subplot(1,2,1)
plt.plot(t_nyquist,f_nyquist,'g.',label='$sample \, points$')
plt.plot(t_long, f_long, 'g-',label='$interpolant$')
#plt.ylim([min(f), max(f)])
plt.legend()