# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 01:16:35 2016

@author: Kedar
"""

import math
import pylab as plt
from time_spectral import myLinspace
from matplotlib import animation         # for specifying the writer
import numpy as np

# frequencies in the underlying signal
actual_freqs = [1.0, 2.3]

# frequencies input by the user to the HB method
freqs = [1.0, 2.3]



#-----------------------------------------------------------------------------#
def period_given_freqs(freqs):
    '''
    if the frequencies that make up your solution are all rational numbers, the
    resulting signal is a periodic function. given that set of frequencies, 
    this function will return the corresponding period.
    '''
    import decimal
    from functools import reduce
    # find the maximum number of decimal points seen the given frequencies
    decimal_pts = [-decimal.Decimal(str(f)).as_tuple().exponent for f in freqs]
    max_dec_pts = max(decimal_pts)
    # multiply all frequencies by the power of ten that make them integers
    scaled_freqs = [int(freq*pow(10,max_dec_pts)) for freq in freqs]
    # find the greatest common divisor of these scaled frequencies
    GCD = reduce(lambda x,y: math.gcd(x,y), scaled_freqs)
    # compute the multiple of the lowest-frequency period
    lowest_multiple = min(scaled_freqs)/GCD
    lowest_freq_period = 2.0*math.pi/min(freqs)
    # find the total period
    total_period = lowest_freq_period*lowest_multiple
    return total_period
#-----------------------------------------------------------------------------#
def my_non_periodic_fun(t, freqs):
    '''
    given a set of specfied frequencies and time points, this subroutine 
    samples a function composed of a sum of sinusoids oscillating at the 
    frequencies specified
    '''
    f = []
    df = []
    # run through all the time samples
    for t_i in t:
        f_i = 0
        df_i = 0
        # run through all the frequencies
        for freq in freqs:
            f_i += math.sin(freq*t_i)
            df_i += freq*math.cos(freq*t_i)
        f.append(f_i)
        df.append(df_i)
    return f, df
#-----------------------------------------------------------------------------#
def my_non_periodic_ode(t, u, freqs):
    
    '''
    given a set of specfied frequencies and time points, this subroutine 
    samples a function (meant the represent the RHS of an ODE) composed of a 
    sum of sinusoids oscillating at the frequencies specified
    '''
    import math
    
    # The equation for periodic population harvesting
    C = 10                    # carrying capacity
    # expression giving the derivative    
    dudt = u*(1-(u/C))
    # run through all the frequencies
    for freq in freqs:
        dudt = dudt - math.sin(freq*t)
    return dudt
#-----------------------------------------------------------------------------#
def harmonic_balance_operator(freqs):
    '''
    given a discrete set of frequencies, compute the harmonic-balance 
    differential operator matrix
    '''
    import numpy as np
    from functools import reduce
    # set the number of the time instances
    K = len(freqs)
    N = 2*K+1
    # time period corresponding to the lowest frequency
    T_lowest_freq = (2.0*math.pi)/min(freqs)
    # set the location of the time instances 
    delta_t = T_lowest_freq/N
    t_HB = np.array([i*delta_t for i in range(N)])
    # create a list of all N discrete frequencies
    # f = [0, f_1, ..., f_K, -f_K, ..., -f_1]
    f = [0]+freqs+[-freqs[-(i+1)] for i in range(K)]
    # create the diagonal matrix holding the frequencies
    f_imag = [1j*f_i for f_i in f]
    D = np.diag(f_imag)
    # create the forward-transform matrix, E
    E_inv = np.zeros([N,N], dtype=np.complex_)
    for i in range(N):
        for j in range(N):
            E_inv[i][j] = np.exp(1j*f[j]*t_HB[i])
    # take the inverse of the matrix E
    E = np.linalg.inv(E_inv)
    # compute the operator (D_HB = E_inv*D*E)
    # is always real...why?
    D_HB = np.real(reduce(lambda x,y: np.dot(x,y), [E_inv, D, E]))
    # return the operator matrix and the time instances
    return (D_HB, t_HB)
#-----------------------------------------------------------------------------#
# this function interpolates a series of points with a Fourier series #########
def fourierInterp_given_freqs(x, y, freqs, x_int=None):
    
    """
    This function interpolates a given set of ordinates and abscissas with a
    Fourier series. The function returns a Fourier interpolation (of the 
    highest degree trig polynomial allowed by the Nyquist Criterion) on the 
    given grid of new abscissas. If no vector of desired abscissas is given, 
    the set of interpolant abscissas is set automatically to be ten times as 
    finely spaced as the original. The first derivative of the interpolant is 
    also returned. Note that the interpolants will only be exact if the given 
    points are just one shy of representing an exact period.
    
    Input:
      - abscissas, x (as a list) (leave out last, duplicate point in period)
      - ordinates, y (as a list) (again, leave out last point, if periodic)
      - new abscissas, x_int (as a list) (optional! defaults to 10x refinement)
    Output:
      - new abscissas, x_int (as a list)
      - interpolated ordinates, y_int (as a list)
      - derivative of the interpolant, dydx_int (as a list)
    """
    
    import math
    
    # refinment factor for the interpolant. (If there are originally 10 points
    # but you want the Fourier Series to be defined on 50 points, then the 
    # refinement factor is 5.)
    refine_fac = 10
    
    # preliminaries
    n = len(x)                  # number of abscissas
    n = float(n)                # for mathematical clarity in the next line
    m = math.floor((n-1)/2)     # highest degree trig polynomial allowed by NC
    
    # establish scaling to the domain [0,2pi)
    x_interval = x[1]-x[0]      # even interval in abscissas
    period = n*x_interval       # extrapolate and find length of the period
    
    # initalization
    n = int(n)          # recast as int
    m = int(m)          # recast as int
    a = [0.0]*(m+1)       # "+1" in order to incorporate "DC value" at a_0
    b = [0.0]*(m+1)       # b_0 never used, but included to match length of a
    
    # compute the coefficients
    index = 0
    for j in freqs:
        a[index] = 0.0
        b[index] = 0.0
        for i in range(n):
            scaled_x = (2*math.pi/period)*(x[i]-x[0])
            a[index] += (2.0/n)*math.cos(j*scaled_x)*y[i]
            b[index] += (2.0/n)*math.sin(j*scaled_x)*y[i]
        index += 1
        
    # set x_int, if it hasn't been given
    if x_int == None:
        n_int = refine_fac*(n)
        x_int = myLinspace(x[0],x[-1]+x_interval,n_int)
    else:
        n_int = len(x_int)
    
    # find the actual interpolation
    y_int = [0.0]*n_int
    dydx_int = [0.0]*n_int
    for i in range(n_int):
        y_int[i] = a[0]/2.0    # the "DC" value
        dydx_int[i] = 0.0      # intialize the summation
        scaled_x_int = (2*math.pi/period)*(x_int[i]-x[0])
        index = 0
        for j in freqs:
            y_int[i] += a[index+1]*math.cos((j+1)*scaled_x_int) + \
                        b[index+1]*math.sin((j+1)*scaled_x_int)
            dydx_int[i] += (2*math.pi/period)*(j+1)* \
                           (b[index+1]*math.cos((j+1)*scaled_x_int) - \
                            a[index+1]*math.sin((j+1)*scaled_x_int))
            index += 1
    return (x_int, y_int, dydx_int)
    
    
    
    
# compute and print the periods of the various signals
T_HB_sol = period_given_freqs(freqs)
T_actual_sol = period_given_freqs(actual_freqs)
print('\nPeriod of HB solution:', round(T_HB_sol,3))
print('\nPeriod of the ODE solution:', round(T_actual_sol,3),'\n')

###############################################################################
# [harmonic balance] preliminaries ############################################
###############################################################################

# create the harmonic balance operator matrix
D_HB, t_HB = harmonic_balance_operator(freqs)
print('D_HB = ',D_HB)

###############################################################################
# [harmonic balance] Check to see if Harmonic Balance operator is working #####
###############################################################################

# [exact] time period corresponding to the lowest frequency being used for HB
T_lowest_freq = (2.0*math.pi)/min(freqs)
# [exact] discretize that longest time period
points_per_unit_time = 100
t = myLinspace(0, T_lowest_freq,points_per_unit_time*math.ceil(T_lowest_freq))
f,df = my_non_periodic_fun(t, actual_freqs)


# [HB] checking to see if we can find time derivatives
f_HB, dummy = my_non_periodic_fun(t_HB, actual_freqs)
# turn it into a numpy column vector
f_HB = np.array(f_HB)
f_HB = f_HB.reshape(f_HB.size,1)
df_HB = np.dot(D_HB, f_HB)

# plot everything
plt.close('all')
plt.figure()
plt.plot(t, f, label='$f_{exact}$')
plt.plot(t_HB, f_HB, 'ko', label='$f_{HB}$')
plt.plot(t, df, 'r--', label='$df/dt_{exact}$')
plt.plot(t_HB, df_HB, 'go', label='$df/dt_{HB}$')

print('t = ', np.array(t_HB))
print('f = ', f_HB)

t_HB_int,df_HB_int,dummy = fourierInterp_given_freqs(t_HB,df_HB,freqs)
plt.plot(t_HB_int,df_HB_int, 'g--')
plt.xlabel('$t$', fontsize=16)
plt.ylabel('$f(t)$', fontsize=16)
plt.legend(loc='best')

###########################################################################
# [time accurate] explicit euler ##########################################
###########################################################################
delta_t = 0.05             # time step
initial_value = 2.0        # intitial condition

t_start = 0                # initial time
t_end = 100                 # approx. final time (stop at or just after)
  
f = []
times = []
time_points = int(math.ceil((t_end-t_start)/delta_t)+1)

# time stepping
for n in range(time_points):
    times.append(t_start+n*delta_t)
    if n == 0:
        # record the initial condition
        f.append(initial_value)
    else:
        # explicitly step forward in time (give it the actual frequencies!)
        f.append(f[n-1] + delta_t*my_non_periodic_ode(times[n-1],f[n-1],actual_freqs))

# plotting: USER INPUTS! do you want to animate the solution history or just
# plot the final result? (True = animate, False = just print final result)
animate_plot = False                
plot_name = 'time-accurate ODE (HB)'
n_images = time_points            # total number of images computed
skip_images = 15                   # images to skip between animation frames

# plotting: initializations
fig = plt.figure()
# plotting: things that will not be changing inside the loop
plt.xlabel('$t$', fontsize=18)
plt.ylabel('$f(t)$', fontsize=18)
plt.xlim(0,t_end)
vertical_padding = (max(f)-min(f))/4.0
plt.ylim(min(f)-vertical_padding,max(f)+vertical_padding)
title = ''
counter=1
for freq in freqs:
    title = title + '$\omega_{'+str(counter)+'} ='+str(freq)+'\quad $'
    counter += 1
plt.title(title+'$\Delta t = \,$'+str(delta_t))
# plotting: set the total number of frames
if animate_plot == True:
    # capture all frames (skipping, if necessary) and the final frame
    all_frames = list(range(0,n_images,skip_images+1))+[n_images-1]
else:
    # no animation: just capture the last one
    all_frames = [n_images-1]
# plotting: capturing the movie
writer = animation.writers['ffmpeg'](fps=15)
with writer.saving(fig, plot_name+'.mp4', 300):
    frame = 0
    for n in all_frames:
        plt.plot(times[:n+1],f[:n+1],'k-')
        # progress monitor
        percent_done = float(n)*100.0/(n_images-1)
        print('capturing fig. '+plot_name+' (frame #'+str(frame)+'): ', \
               round(percent_done,2),'%')
        writer.grab_frame()
        frame += 1
    writer.grab_frame()
# plotting: save an image of the final frame
print('\n'+'saving final image...')
plt.savefig(plot_name, dpi=500)
print('figure saved: ' + plot_name)
# free memory used for the plot
plt.close(fig)