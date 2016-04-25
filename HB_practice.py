# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 01:16:35 2016

@author: Kedar
"""

import math
import pylab as plt
from time_spectral import myLinspace, myNorm
from matplotlib import animation         # for specifying the writer
import numpy as np
plt.close('all')

# angular frequencies in the underlying signal
actual_omegas = [1.0, 3.0]

# angular frequencies input by the user to the HB method
omegas = [1.0, 3.0]
omegas = actual_omegas


def myfloat(number):
    from decimal import Decimal
    pts_after = -Decimal(str(number)).as_tuple().exponent
    return float(int(number*10**pts_after))/10**pts_after

#-----------------------------------------------------------------------------#
def period_given_freqs(omegas):
    '''
    if the frequencies that make up your solution are all rational numbers, the
    resulting signal is a periodic function. given that set of frequencies, 
    this function will return the corresponding period.
    '''
    import decimal
    from functools import reduce
    # find the maximum number of decimal points seen the given frequencies
    decimal_pts = [-decimal.Decimal(str(f)).as_tuple().exponent for f in omegas]
    max_dec_pts = max(decimal_pts)
    # multiply all frequencies by the power of ten that make them integers
    scaled_omegas = [int(omega*pow(10,max_dec_pts)) for omega in omegas]
    # find the greatest common divisor of these scaled frequencies
    GCD = reduce(lambda x,y: math.gcd(x,y), scaled_omegas)
    # compute the multiple of the lowest-frequency period
    lowest_multiple = min(scaled_omegas)/GCD
    lowest_omega_period = 2.0*math.pi/min(omegas)
    # find the total period
    total_period = lowest_omega_period*lowest_multiple
    return total_period
#-----------------------------------------------------------------------------#
def my_non_periodic_fun(t, omegas):
    '''
    given a set of specfied frequencies and time points, this subroutine 
    samples a function composed of a sum of sinusoids oscillating at the 
    frequencies specified
    returns numpy column vectors
    '''
    import numpy as np
    f = []
    df = []
    # run through all the time samples
    for t_i in t:
        f_i = 0 + 5
        df_i = 0
        # run through all the frequencies
        for omega in omegas:
            f_i += math.sin(omega*t_i)
            df_i += omega*math.cos(omega*t_i)
        f.append(f_i)
        df.append(df_i)
    # turn f and df into numpy column vectors
    f = np.array(f)
    f = f.reshape(f.size,1)
    df = np.array(df)
    df = df.reshape(df.size,1)
    return f, df
#-----------------------------------------------------------------------------#
def my_non_periodic_ode(t, u, omegas):
    
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
    for omega in omegas:
        dudt = dudt - math.sin(omega*t)
    return dudt
#-----------------------------------------------------------------------------#
def harmonic_balance_operator(omegas):
    '''
    given a discrete set of frequencies, compute the harmonic-balance 
    differential operator matrix
    '''
    import numpy as np
    from functools import reduce
    # set the number of the time instances
    K = len(omegas)
    N = 2*K+1
    # time period corresponding to the lowest frequency
    T_lowest_omega = (2.0*math.pi)/min(omegas)
    # set the location of the time instances 
    delta_t = T_lowest_omega/N
    t_HB = np.array([i*delta_t for i in range(N)])
    # create a list of all N discrete frequencies
    # w = [0, w_1, ..., w_K, -w_K, ..., -w_1]
    w = [0]+omegas+[-omegas[-(i+1)] for i in range(K)]
    # create the diagonal matrix holding the frequencies
    w_imag = [1j*w_i for w_i in w]
    C = np.diag(w_imag)
    # create the forward-transform matrix, E
    E_inv = np.zeros([N,N], dtype=np.complex_)
    for i in range(N):
        for j in range(N):
            E_inv[i][j] = (1/N)*np.exp(1j*w[j]*t_HB[i])
    # take the inverse of the matrix E
    E = np.linalg.inv(E_inv)
    # compute the operator (D_HB = E_inv*C*E)
    # is always real...why?
    D_HB = np.real(reduce(lambda x,y: np.dot(x,y), [E_inv, C, E]))
    # return the operator matrix and the time instances
    return (D_HB, t_HB)
#-----------------------------------------------------------------------------#
def fourierInterp_given_freqs(x, y, omegas, x_int=None):
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
    N = len(x)
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
    # create separate lists for the cosine and sine coefficients
    a = []
    b = [0]
    for i in range(N):
        if i==0 or i%2==1:
            a.append(Fourier_coeffs[i][0])
        else:
            b.append(Fourier_coeffs[i][0])
    # set x_int, if it hasn't been given
    if x_int == None:
        n_int = refine_fac*(n)
        x_int = myLinspace(x[0],x[-1]+x[1],n_int)
    else:
        n_int = len(x_int)
    # find the actual interpolation
    y_int = [0.0]*n_int
    dydx_int = [0.0]*n_int
    for i in range(n_int):
        y_int[i] = a[0]/2.0    # the "DC" value
        dydx_int[i] = 0.0      # intialize the summation
        for j in range(m-1):
            y_int[i] += a[j+1]*math.cos(omegas[j+1]*x_int[i]) + \
                        b[j+1]*math.sin(omegas[j+1]*x_int[i])
            dydx_int[i] += omegas[j+1]* \
                           (b[j+1]*math.cos(omegas[j+1]*x_int[i]) - \
                            a[j+1]*math.sin(omegas[j+1]*x_int[i]))
    return (x_int, y_int, dydx_int)
#-----------------------------------------------------------------------------#    
def plot_eigenvalues(A):
    '''
    Plots the eigenvalues of the given matrix on the complex plane.
    '''
    import numpy as np
    import pylab as plt
    
    eig_result = np.linalg.eig(A)
    eigs = eig_result[0]
    Re_parts = np.real(eigs)
    Im_parts = np.imag(eigs)
    plt.plot(Re_parts,Im_parts,'ko')
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.axis('equal')
    plt.title('$eig\\left(D_{HB}\\right)$')
    plt.grid()
#-----------------------------------------------------------------------------#
#######################################
# actual periods of the two solutions #
#######################################

# compute and print the periods of the various signals
T_HB_sol = period_given_freqs(omegas)
T_actual_sol = period_given_freqs(actual_omegas)
print('\nPeriod of HB solution:', round(T_HB_sol,3))
print('\nPeriod of the ODE solution:', round(T_actual_sol,3),'\n')

###########################################################################
# [harmonic balance] Check to see if Harmonic Balance operator is working #
###########################################################################

# [exact] time period corresponding to the lowest frequency being used for HB
T_lowest_omega = (2.0*math.pi)/min(omegas)
# [exact] discretize that longest time period
points_per_unit_time = 100
t = myLinspace(0, T_lowest_omega,points_per_unit_time*math.ceil(T_lowest_omega))
f,df = my_non_periodic_fun(t, actual_omegas)

# create the harmonic balance operator matrix
D_HB, t_HB = harmonic_balance_operator(omegas)
print('omegas = ', omegas)
print('D_HB = ', D_HB,'\n')
print('cond(D_HB) =', np.linalg.cond(D_HB))
plot_eigenvalues(D_HB)

# [HB] checking to see if we can find time derivatives
f_HB, dummy = my_non_periodic_fun(t_HB, actual_omegas)
# multiply by the HB operator matrix 
df_HB = np.dot(D_HB, f_HB)

# plot everything
plt.figure()
plt.plot(t, f, label='$f_{exact}$')
plt.plot(t, df, 'r-', label='$df/dt_{exact}$')
plt.plot(t_HB, f_HB, 'ko', label='$f_{HB}$')
plt.plot(t_HB, df_HB, 'go', label='$df/dt_{HB}$')
t_HB_int, df_HB_int, dummy = fourierInterp_given_freqs(t_HB,df_HB,omegas)
plt.plot(t_HB_int,df_HB_int, 'g--', label='$spectral\,\,interp.$')
plt.xlabel('$t$', fontsize=16)
plt.ylabel('$f(t)$', fontsize=16)
plt.legend(loc='best')
plt.title('$\omega_{actual} = \{'+str(actual_omegas)[1:-1]+'\} \quad\quad \omega_{used} = \{'+str(omegas)[1:-1]+'\}$', fontsize=16)

############################################################################
# See how the error in the derivatives changes if the freqs used are wrong #
############################################################################

# set the error range to study
abs_percent_error = 75

# initialize the set of percent errors to try 
percent_errors = myLinspace(-abs_percent_error, abs_percent_error, 2*abs_percent_error+1)
# instantiate list for plotting
max_norm_sol_error = 0
norm_sol_errors = []
# find the interpolated HB derivative with the correct angular frequencies
D_HB, t_HB_actual = harmonic_balance_operator(actual_omegas)
f_HB_actual, dummy = my_non_periodic_fun(t_HB, actual_omegas)
df_HB = np.dot(D_HB, f_HB_actual)
t_HB_int_actual, df_HB_int_actual, dummy = fourierInterp_given_freqs(t_HB, df_HB, actual_omegas)
# loop through the various errors to try
for error_percentage in percent_errors:
    # set the wrong angular frequencies
    omegas = [(1.0+error_percentage/100)*omega for omega in actual_omegas]
    # find the interpolated HB derivative with the incorrect frequencies
    D_HB, t_HB = harmonic_balance_operator(omegas)
    f_HB, dummy = my_non_periodic_fun(t_HB, actual_omegas)
    df_HB = np.dot(D_HB, f_HB)
    t_HB_int_wrong, df_HB_int_wrong, dummy = fourierInterp_given_freqs(t_HB,df_HB,omegas)
    # compute error of incorrect solution, using values at interpolation points
    # just just the error in df
    sol_error = [(df_HB_int_wrong[i]-df_HB_int_actual[i])/df_HB_int_actual[i] for i in range(len(df_HB_int_actual))]
    # compute distance on (t,df) plane    
    sol_error = [math.sqrt((df_HB_int_wrong[i]-df_HB_int_actual[i])**2 + (t_HB_int_wrong[i]-t_HB_int_actual[i])**2) for i in range(len(df_HB_int_actual))]
    
    # find the l2-norm of the error
    norm_sol_error = myNorm(sol_error)
    norm_sol_errors.append(norm_sol_error)
    # store the current omegas if they yield a higher error
    if max_norm_sol_error < norm_sol_error:
        max_norm_sol_error = norm_sol_error
        max_error_omegas = omegas
# plot the norm of the solution errors
plt.figure()
plt.plot(percent_errors, norm_sol_errors, 'k.-')
plt.xlabel('$\% \, error \; of \; each \; \omega_{i}$', fontsize=16)
#plt.ylabel('$\|\\frac{\\frac{\partial f}{\partial t}_{wrong}-\\frac{\partial f}{\partial t}_{actual}}{\\frac{\partial f}{\partial t}_{actual}}\|_2$', fontsize=26)
plt.ylabel('$\\left\Vert \sqrt{ \\left( \\frac{\partial f}{\partial t}_{wrong}-\\frac{\partial f}{\partial t}_{actual} \\right)^2 + \\left( t_{wrong}-t_{actual} \\right)^2}\\right\Vert_2$', fontsize=16)
plt.title('$\omega_{actual} = \{'+str(actual_omegas)[1:-1]+'\}$')
plt.tight_layout()
# find the interpolated HB derivative with the most incorrect frequencies
omegas = max_error_omegas
D_HB, t_HB = harmonic_balance_operator(omegas)
f_HB, dummy = my_non_periodic_fun(t_HB, actual_omegas)
df_HB_wrong = np.dot(D_HB, f_HB)
t_HB_int, df_HB_int_wrong, dummy = fourierInterp_given_freqs(t_HB,df_HB,omegas)
# plot the correct derivative and the worst answer studied
plt.figure()
plt.plot(t, f, label='$f_{exact}$')
plt.plot(t_HB_actual, f_HB_actual, 'ko', label='$f_{HB}$')
plt.plot(t, df, 'r-', label='$df/dt_{exact}$')
plt.plot(t_HB, df_HB_wrong, 'go', label='$df/dt_{HB,wrong}$')
t_HB_int, df_HB_int, dummy = fourierInterp_given_freqs(t_HB,df_HB_wrong,omegas)
plt.plot(t_HB_int,df_HB_int, 'g--', label='$spectral\,\,interp.$')
plt.xlabel('$t$', fontsize=16)
plt.ylabel('$f(t)$', fontsize=16)
plt.legend(loc='best')
plt.title('$\omega_{actual} = \{'+str(actual_omegas)[1:-1]+'\} \quad\quad \omega_{used} = \{'+str(omegas)[1:-1]+'\}$', fontsize=16)

#####################################################################
# See how condition number varies with selected angular frequencies #
#####################################################################
first_omega = 1.0
starting_multiple = 1
max_multiple = 5
multiples = myLinspace(starting_multiple, max_multiple, 150*int(max_multiple-starting_multiple)+1)
plot_nondim_omegas = False

conds = []
nondim_omegas = []
for multiple in multiples:
    second_omega = first_omega*multiple
    freqs = [first_omega, second_omega]
    nondim_omegas.append(2.0*(second_omega-first_omega)/(second_omega+first_omega))
    D_HB, t_HB = harmonic_balance_operator(freqs)
    current_cond = np.linalg.cond(D_HB)    
    conds.append(current_cond)
    #print('multiple: '+str(multiple)+'\tcond: '+str(current_cond))
peaks = [float("{0:.3f}".format(multiples[i])) for i in range(len(conds)) if conds[i] > np.average(conds)+2*np.std(conds)]
nondim_peaks = [float("{0:.3f}".format(nondim_omegas[i])) for i in range(len(conds)) if conds[i] > np.average(conds)+2*np.std(conds)]

plt.figure()
if plot_nondim_omegas:
    plt.semilogy(nondim_omegas, conds, 'k.-')
    plt.xlabel('$\delta^*_{\omega_2}=2\,\\frac{\omega_2-\omega_1}{\omega_2+\omega_1}$', fontsize=16)
    plt.title('$outliers \,\, at: \quad '+str(nondim_peaks)[1:-1]+'$', fontsize=18)
else:
    plt.semilogy(multiples, conds, 'k.-')
    plt.xlabel('$\omega_2/\omega_1$', fontsize=16)
    plt.title('$outliers \,\, at: \quad '+str(peaks)[1:-1]+'$', fontsize=18)
plt.ylabel('$\kappa(D_{HB})$', fontsize=16)

##################################
# [time accurate] explicit euler #
##################################
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
        f.append(f[n-1] + delta_t*my_non_periodic_ode(times[n-1],f[n-1],actual_omegas))

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
for omega in omegas:
    title = title + '$\omega_{'+str(counter)+'} ='+str(omega)+'\quad $'
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