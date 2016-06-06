# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 01:16:35 2016
@author: Kedar
"""

import math
from matplotlib import pyplot as plt
from time_spectral import myLinspace, myNorm, linearInterp
from matplotlib import animation         # for specifying the writer
import numpy as np
import webbrowser
import copy

# turn off interactive mode, so that plot windows don't pop up
plt.ioff()

#-----------------------------------------------------------------------------#
def period_given_freqs(omegas):
    '''
    if the frequencies that make up your solution are all rational numbers, the
    resulting signal is a periodic function. given that set of frequencies, 
    this function will return the corresponding period.
    '''
    import decimal
    from functools import reduce
    from math import gcd
    # find the maximum number of decimal points seen the given frequencies
    decimal_pts = [-decimal.Decimal(str(f)).as_tuple().exponent for f in omegas]
    max_dec_pts = max(decimal_pts)
    # multiply all frequencies by the power of ten that make them integers
    scaled_omegas = [int(omega*pow(10,max_dec_pts)) for omega in omegas]
    # find the greatest common divisor of these scaled frequencies
    GCD = reduce(lambda x,y: gcd(x,y), scaled_omegas)
    # find the total period
    total_period = 2.0*math.pi/(GCD/pow(10,max_dec_pts))
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
    sum of sinusoids oscillating at the frequencies specified. returns a numpy 
    array of the samples of the RHS of the ODE.
    '''
    import numpy as np
    # initialize single-entry flag
    single_entry = False
    # if inputs are floats/ints, convert to lists
    if type(t)==int or type(t)==float:
        t = [t]
        single_entry = True
    if type(u)==int or type(u)==float:
        u = [u]
        single_entry = True
    # instantiate empty list
    dudt = []
    # The equation for periodic population harvesting
    C = 10                    # carrying capacity
    for i in range(len(t)):
        # expression giving the derivative    
        dudt_i = u[i]*(1-(u[i]/C))
        # run through all the frequencies
        for omega in omegas:
            dudt_i -= np.sin(omega*t[i])
        # append to result list
        dudt.append(dudt_i)
    if single_entry:
        dudt = float(dudt[0])
    else:
        # convert to a numpy array
        dudt = np.array(dudt).reshape(len(dudt),1)
    return dudt
#-----------------------------------------------------------------------------#
def HB_time_instances(omegas, time_discretization='use_Nyquist'):
    '''
    this function returns a numpy array corresponding to the desired time
    discretization:
      - time_discretization = 'use_T1' (equally spaced points just spanning the
                                  period corresponding to the lowest frequency)
      - time_discretization = 'use_Nyquist' (use Nyquist criterion for spacing)
    '''
    import numpy as np
    # set the number of the time instances
    K = len(omegas)
    N = 2*K+1           # required for Fourier interpolation
    if time_discretization == 'use_T1':
        # time period corresponding to the lowest frequency
        T_lowest_omega = (2.0*np.pi)/min(omegas)
        # compute the time interval
        delta_t = T_lowest_omega/N
    if time_discretization == 'use_Nyquist':
        # scaling up nyquist frequency by this factor (must be > 1)
        scaling_fac = 1.1
        # nyquist rate (this is a bandlimited signal)
        nyquist_rate = max(omegas)/np.pi
        # find the corresponding time interval
        delta_t = 1.0/(scaling_fac*nyquist_rate)
    # set the location of the time instances
    t_HB = np.array([i*delta_t for i in range(N)])
    return t_HB
#-----------------------------------------------------------------------------#
def HB_forward_transform_matrix(omegas, time_discretization):
    '''
    given the discrete set of nonzero angular frequencies and the time
    instances of interest, this subroutine returns the "E inverse" matrix that 
    represents the forward discrete Fourier transform taken at the specific 
    frequencies
    '''
    import numpy as np
    # set the location of the time instances
    t_HB = HB_time_instances(omegas, time_discretization)
    # set the number of the time instances
    K = len(omegas)
    N = 2*K+1           # required for Fourier interpolation
    # create a list of all N discrete frequencies
    # w = [0, w_1, ..., w_K, -w_K, ..., -w_1]
    w = [0]+omegas+[-omegas[-(i+1)] for i in range(K)]
    # create the forward-transform matrix, E_inv
    E_inv = np.zeros([N,N], dtype=np.complex_)
    for i in range(N):
        for j in range(N):
            E_inv[i][j] = (1/N)*np.exp(1j*w[j]*t_HB[i])
    return E_inv
#-----------------------------------------------------------------------------#
def harmonic_balance_operator(omegas, time_discretization='use_Nyquist'):
    '''
    given a discrete set of frequencies, compute the harmonic-balance 
    differential operator matrix
    '''
    import numpy as np
    from functools import reduce
    # set the location of the time instances
    t_HB = HB_time_instances(omegas, time_discretization)
    # create a list of all N discrete frequencies
    # w = [0, w_1, ..., w_K, -w_K, ..., -w_1]
    w = [0]+omegas+[-omegas[-(i+1)] for i in range(len(omegas))]
    # create the diagonal matrix holding the frequencies
    w_imag = [1j*w_i for w_i in w]
    C = np.diag(w_imag)
    # construct the forward-tranform matrix
    E_inv = HB_forward_transform_matrix(omegas, time_discretization)
    # take the inverse of the matrix E
    E = np.linalg.inv(E_inv)
    # compute the operator (D_HB = E_inv*C*E)
    # is always real...why?
    D_HB = np.real(reduce(lambda x,y: np.dot(x,y), [E_inv, C, E]))
    # return the operator matrix and the time instances
    return (D_HB, t_HB)
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
    if type(x_int) == bool:
        n_int = refine_fac*(n)
        x_int = myLinspace(x[0],x[-1]+x[1],n_int)
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
def plot_eigenvalues(A, auto_open=False):
    '''
    Plots the eigenvalues of the given matrix on the complex plane.
    Open the plot automatically, if desired.
    '''
    import numpy as np
    from matplotlib import pyplot as plt
    import webbrowser
    # compute the eigenvalues and eigenvectors
    eig_result = np.linalg.eig(A)
    # extract the eigenvalues
    eigs = eig_result[0]
    # separate the real and imaginary parts
    Re_parts = np.real(eigs)
    Im_parts = np.imag(eigs)
    # plot eigenvalues on the complex plane
    plot_name = 'HB_eigenvalues'
    plt.figure(plot_name)
    plt.plot(Re_parts,Im_parts,'ko')
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.axis('equal')
    decimals_to_print = 4
    title = ''
    for k in range(len(eigs)):
        if abs(Re_parts[k]) != 0.0:
            title += str(np.around(Re_parts[k],decimals_to_print))
        if abs(Im_parts[k]) != 0.0:
            if np.sign(Im_parts[k]) == 1:
                title += '\plus'
            title += str(np.around(Im_parts[k],decimals_to_print))+'i'
        if not k == len(eigs)-1:
            title += ',\quad'
    plt.title('$eig\\left(D_{HB}\\right) \, = \,'+title+'$', y=1.03)
    plt.grid()
    # save plot and close
    print('\n\t'+'saving final image...', end='')
    file_name = plot_name+'.png'
    plt.savefig(file_name, dpi=300)
    print('figure saved: '+plot_name)
    plt.close(plot_name)
    # open the saved image, if desired
    if auto_open:
        webbrowser.open(file_name)
#-----------------------------------------------------------------------------#
def compute_cost(t_HB, f_HB, omegas, delta_t, actual_omegas):
    '''
    this subroutine accepts a partially converged harmonic-balance solution and
    the set of angular frequencies asssumed while computing that solution. each
    time instance along the solution is then taken to be the initial 
    condition of the new time-accurate problem, from which we take one Euler
    step and note the new position that is reached. the HB solution points are
    then "fast-forwarded" (moved ahead in time) by the "long" period that 
    corresponds to the given set of frequencies. now, we take these new points
    as initial conditions and, like before, take one time-accurate step and 
    note the new positions. subtract these new points from the corresponding
    time-accurately-stepped positions from the the original solution. take the
    l^2 norm of the differences. return this cost.
    '''
    import numpy as np
    # evaluate the differential equation at the given solution points
    func_evaluations = my_non_periodic_ode(t_HB,f_HB,actual_omegas)
    # take one time-accurate step at each time instance
    f_HB_plus_dt = f_HB + delta_t*func_evaluations
    # "long" period corresponding to the given set of frequencies
    T_omega_star = period_given_freqs(omegas)
    # "fast forward" current time instances ahead by one "long" period
    t_HB_plus_T = t_HB + T_omega_star*np.ones(len(t_HB))
    # evaluate the differential equation at the fast-forwarded instances, 
    # and the HB solution values as initial conditions
    func_evaluations_plus_T = my_non_periodic_ode(t_HB_plus_T,f_HB,actual_omegas)
    # take one time-accurate step from these "fast-forwarded" points
    f_HB_plus_T_plus_dt = f_HB + delta_t*func_evaluations_plus_T
    # evaluate the cost function    
    cost = np.linalg.norm(f_HB_plus_T_plus_dt-f_HB_plus_dt)
    return cost
#-----------------------------------------------------------------------------#

###############
# user inputs #
###############

# angular frequencies in the underlying signal
actual_omegas = [1.0, 2.5]
actual_omegas = [1.0, 2.1, 6.0]
#actual_omegas = [9.8, 24.5]
#actual_omegas = [9.8, 39.2]
# angular frequencies input by the user to the HB method
omegas = [1.0, 3.0]
omegas = copy.copy(actual_omegas)

# select the time discretization to use for specifying the of time instances
time_discretization = 'use_Nyquist'
#time_discretization = 'use_T1'

#######################################
# actual periods of the two solutions #
#######################################

# compute and print the periods of the various signals
T_HB_sol = period_given_freqs(omegas)
T_actual_sol = period_given_freqs(actual_omegas)

###########################################################################
# [harmonic balance] Check to see if Harmonic Balance operator is working #
###########################################################################

# print message to the screen
print('\nchecking the properties of the HB operator matrix...')

# "exact" fine time grid to evaluate function
T_lowest_omega = 2.0*np.pi/min(omegas)
t_fine_points = 200
t = np.linspace(0.0,T_lowest_omega, t_fine_points)
# find the exact function and function derivative values at the time points
f,df = my_non_periodic_fun(t, actual_omegas)

# create the harmonic balance operator matrix
print('\n( time_discretization = ', time_discretization,')')
D_HB, t_HB = harmonic_balance_operator(omegas, time_discretization)

# plot the eigenvalues of the HB operator matrix
plot_eigenvalues(D_HB, auto_open=False)

# [HB] checking to see if we can find time derivatives
f_HB, dummy = my_non_periodic_fun(t_HB, actual_omegas)
# multiply by the HB operator matrix 
df_HB = np.dot(D_HB, f_HB)

# plot everything
plot_name = 'HB_operator_check'
auto_open = False
fourier_interp_HB = True
plt.figure(plot_name)
plt.plot(t, f, label='$f_{exact}$')
plt.plot(t, df, 'r-', label='$df/dt_{exact}$')
plt.plot(t_HB, f_HB, 'ko', label='$f_{HB}$')
if fourier_interp_HB:
    plt.plot(t_HB, df_HB, 'go', label='$df/dt_{HB}$')
    t_HB_int, df_HB_int, dummy = fourierInterp_given_freqs(t_HB,df_HB,omegas,t)
    plt.plot(t_HB_int,df_HB_int, 'g--', label='$spectral\,\,interp.$')
else:
    plt.plot(t_HB, df_HB, 'go--', label='$df/dt_{HB}$')
    dummy = 1
plt.xlabel('$t$', fontsize=16)
plt.ylabel('$f(t)$', fontsize=16)
#plt.ylim(-40,20)
plt.legend(loc='best')
#plt.legend(loc='lower right')
plt.title('$\omega_{actual} = \{'+str(actual_omegas)[1:-1]+'\} \quad\quad \omega_{used} = \{'+str(omegas)[1:-1]+'\}$', fontsize=16)
#plt.title('$\omega_{actual} = \{'+str(actual_omegas)[1:-1]+'\}$', fontsize=16)

# save plot and close
print('\n\t'+'saving final image...', end='')
file_name = plot_name+'.png'
plt.savefig(file_name, dpi=300)
print('figure saved: '+plot_name)
plt.close(plot_name)
# open the saved image, if desired
if auto_open:
    webbrowser.open(file_name)
    
############################################################################
# See how the error in the derivatives changes if the freqs used are wrong #
############################################################################

# set the error range to study
abs_percent_error = 75
# pick which measure of error to use ('f-difference' or 'distance')
error_measure = 'distance'

# print message to the screen
print('\nstudying the effect of supplying incorrect frequencies...')
# initialize the set of percent errors to try 
percent_errors = myLinspace(-abs_percent_error, abs_percent_error, 2*abs_percent_error+1)
# instantiate list for plotting
max_norm_sol_error = 0
norm_sol_errors = []
# find the interpolated HB derivative with the correct angular frequencies
D_HB, t_HB_actual = harmonic_balance_operator(actual_omegas, time_discretization)
f_HB_actual, dummy = my_non_periodic_fun(t_HB, actual_omegas)
df_HB = np.dot(D_HB, f_HB_actual)
t_HB_int_actual, df_HB_int_actual, dummy = fourierInterp_given_freqs(t_HB, df_HB, actual_omegas)
# loop through the various errors to try
for error_percentage in percent_errors:
    # set the wrong angular frequencies
    trial_omegas = [(1.0+error_percentage/100)*omega for omega in actual_omegas]
    # find the interpolated HB derivative with the incorrect frequencies
    D_HB, t_HB = harmonic_balance_operator(trial_omegas, time_discretization)
    f_HB, dummy = my_non_periodic_fun(t_HB, actual_omegas)
    df_HB = np.dot(D_HB, f_HB)
    t_HB_int_wrong, df_HB_int_wrong, dummy = fourierInterp_given_freqs(t_HB,df_HB,trial_omegas)
    # compute error of incorrect solution, using values at interpolation points
    if error_measure == 'f-difference':
        # just just the error in df
        sol_error = [(df_HB_int_wrong[i]-df_HB_int_actual[i])/df_HB_int_actual[i] for i in range(len(df_HB_int_actual))]
    if error_measure == 'distance':
        # compute distance on (t,df) plane    
        sol_error = [math.sqrt((df_HB_int_wrong[i]-df_HB_int_actual[i])**2 + (t_HB_int_wrong[i]-t_HB_int_actual[i])**2) for i in range(len(df_HB_int_actual))]
    # find the l2-norm of the error
    norm_sol_error = myNorm(sol_error)
    norm_sol_errors.append(norm_sol_error)
    # store the current omegas if they yield a higher error
    if max_norm_sol_error < norm_sol_error:
        max_norm_sol_error = norm_sol_error
        max_error_omegas = trial_omegas
        
# plot the norm of the solution errors
plot_name = 'supplying_incorrect_freqs'
auto_open = False
plt.figure(plot_name)
plt.plot(percent_errors, norm_sol_errors, 'k.-')
plt.xlabel('$\% \, error \; of \; each \; \omega_{i}$', fontsize=16)
if error_measure == 'f-difference':
    plt.ylabel('$\|\\frac{\\frac{\partial f}{\partial t}_{wrong}-\\frac{\partial f}{\partial t}_{actual}}{\\frac{\partial f}{\partial t}_{actual}}\|_2$', fontsize=26)
if error_measure == 'distance':
    plt.ylabel('$\\left\Vert \sqrt{ \\left( \\frac{\partial f}{\partial t}_{wrong}-\\frac{\partial f}{\partial t}_{actual} \\right)^2 + \\left( t_{wrong}-t_{actual} \\right)^2}\\right\Vert_2$', fontsize=16)
plt.title('$\omega_{actual} = \{'+str(actual_omegas)[1:-1]+'\}$')
plt.tight_layout()
# save plot and close
print('\n\t'+'saving final image...', end='')
file_name = plot_name+'.png'
plt.savefig(file_name, dpi=300)
print('figure saved: '+plot_name)
plt.close(plot_name)
# open the saved image, if desired
if auto_open:
    webbrowser.open(file_name)
    
# find the interpolated HB derivative with the most incorrect frequencies
max_omegas = max_error_omegas
D_HB, t_HB = harmonic_balance_operator(max_omegas, time_discretization)
f_HB, dummy = my_non_periodic_fun(t_HB, actual_omegas)
df_HB_wrong = np.dot(D_HB, f_HB)
t_HB_int, df_HB_int_wrong, dummy = fourierInterp_given_freqs(t_HB,df_HB,max_omegas)

# plot the correct derivative and the worst answer studied
plot_name = 'result_at_worst_answer'
auto_open = False
plt.figure(plot_name)
plt.plot(t, f, label='$f_{exact}$')
plt.plot(t_HB_actual, f_HB_actual, 'ko', label='$f_{HB}$')
plt.plot(t, df, 'r-', label='$df/dt_{exact}$')
plt.plot(t_HB, df_HB_wrong, 'go', label='$df/dt_{HB,wrong}$')
t_HB_int, df_HB_int, dummy = fourierInterp_given_freqs(t_HB,df_HB_wrong,max_omegas)
plt.plot(t_HB_int,df_HB_int, 'g--', label='$spectral\,\,interp.$')
plt.xlabel('$t$', fontsize=16)
plt.ylabel('$f(t)$', fontsize=16)
plt.legend(loc='best')
plt.title('$\omega_{actual} = \{'+str(actual_omegas)[1:-1]+'\} \quad\quad \omega_{used} = \{'+str(max_omegas)[1:-1]+'\}$', fontsize=16)
# save plot and close
print('\n\t'+'saving final image...', end='')
file_name = plot_name+'.png'
plt.savefig(file_name, dpi=300)
print('figure saved: '+plot_name)
plt.close(plot_name)
# open the saved image, if desired
if auto_open:
    webbrowser.open(file_name)
    
#############################################################################
# See how cond(D_HB) and cond(E_inv) vary with selected angular frequencies #
#############################################################################
first_omega = omegas[0]
starting_multiple = first_omega
penultimate_omega = omegas[-2]
penultimate_multiple = penultimate_omega/first_omega
nondim_penultimate_omega = 2.0*(penultimate_omega-first_omega)/(penultimate_omega+first_omega)
max_multiple = 25.0
multiples = myLinspace(starting_multiple, max_multiple, 50*int(max_multiple-starting_multiple)+1)
plot_nondim_omegas = False

# print message to the screen
print('\nstudying the effect of different frequencies pairs on cond(D_HB) and cond(E_inv)...')
D_conds = []
E_inv_conds = []
average_errors = []
nondim_omegas = []
for multiple in multiples:
    # omegas
    final_omega = first_omega*multiple
    freqs = [first_omega]+omegas[1:-1]+[final_omega]
    nondim_omegas.append(2.0*(final_omega-first_omega)/(final_omega+first_omega))
    # operator    
    D_HB, t_HB = harmonic_balance_operator(freqs, time_discretization)
    current_D_cond = np.linalg.cond(D_HB)
    D_conds.append(current_D_cond)
    # E inverse
    E_inv = HB_forward_transform_matrix(freqs, time_discretization)
    current_E_inv_cond = np.linalg.cond(E_inv)
    E_inv_conds.append(current_E_inv_cond)
    # create a model function using the current pair of freqs and sample this
    # function at the HB time instances
    f_HB, df_exact = my_non_periodic_fun(t_HB, freqs)
    # find the HB derivative
    df_HB = np.dot(D_HB, f_HB)
    # average of the percent errors between the exact derivatives and df_HB
    percent_errors_df = [100.0*(abs(df_HB[i]-df_exact[i]))/abs(df_exact[i]) for i in range(len(f_HB))]
    ave_percent_errors_df = sum(percent_errors_df)/len(percent_errors_df)
    average_errors.append(ave_percent_errors_df)
    #print('multiple: '+str(multiple)+'\tcond: '+str(current_cond))
# find the peaks in these results, defined as being larger than 
D_stdevs_cutoff = 0.0001
D_peaks = [float("{0:.3f}".format(multiples[i])) for i in range(len(D_conds)) if D_conds[i] > np.average(D_conds)+D_stdevs_cutoff*np.std(D_conds)]
D_nondim_peaks = [float("{0:.3f}".format(nondim_omegas[i])) for i in range(len(D_conds)) if D_conds[i] > np.average(D_conds)+D_stdevs_cutoff*np.std(D_conds)]
E_inv_cutoff = 1e12
E_inv_peaks = [float("{0:.3f}".format(multiples[i])) for i in range(len(E_inv_conds)) if E_inv_conds[i] > E_inv_cutoff]
E_inv_nondim_peaks = [float("{0:.3f}".format(nondim_omegas[i])) for i in range(len(E_inv_conds)) if E_inv_conds[i] > E_inv_cutoff]
df_percent_tol = 1e-5
df_error_peaks = [float("{0:.3f}".format(multiples[i])) for i in range(len(average_errors)) if average_errors[i] > df_percent_tol]
df_error_nondim_peaks = [float("{0:.3f}".format(nondim_omegas[i])) for i in range(len(average_errors)) if average_errors[i] > df_percent_tol]

# plot the D_HB result
plot_name = 'cond(D_HB)_vs_freq_ratio'
auto_open = False
plt.figure(plot_name)
if plot_nondim_omegas:
    plt.semilogy(nondim_omegas, D_conds, 'k.-')
    plt.semilogy([nondim_penultimate_omega]*2,[min(D_conds),max(D_conds)],'y--')
    plt.xlabel('$\delta^*_{\omega_K}=2\,\\frac{\omega_K-\omega_1}{\omega_K+\omega_1}$', fontsize=14)
    plt.title('$outliers \,\, at: \,\, '+str(D_nondim_peaks)[1:-1]+'$', fontsize=8)
else:
    plt.semilogy(multiples, D_conds, 'k.-')
    plt.semilogy([penultimate_omega]*2,[min(D_conds),max(D_conds)],'y--')
    plt.xlabel('$\omega_K/\omega_1$', fontsize=14)
    plt.title('$outliers \,\, at: \,\, '+str(D_peaks)[1:-1]+'$', fontsize=8)
plt.ylabel('$\kappa(D_{HB})$', fontsize=16)
#plt.tight_layout()
# save plot and close
print('\n\t'+'saving final image...', end='')
file_name = plot_name+'.png'
plt.savefig(file_name, dpi=300)
print('figure saved: '+plot_name)
plt.close(plot_name)
# open the saved image, if desired
if auto_open:
    webbrowser.open(file_name)
    
# plot the E_inv result
plot_name = 'cond(E_inv)_vs_freq_ratio'
auto_open = False
plt.figure(plot_name)
if plot_nondim_omegas:
    plt.semilogy(nondim_omegas, E_inv_conds, 'k.-')
    plt.semilogy([nondim_penultimate_omega]*2,[min(E_inv_conds),max(E_inv_conds)],'y--')
    plt.xlabel('$\delta^*_{\omega_K}=2\,\\frac{\omega_K-\omega_1}{\omega_K+\omega_1}$', fontsize=14)
    #plt.title('$outliers \,\, at: \,\, '+str(E_inv_nondim_peaks)[1:-1]+'$', fontsize=6)
    plt.ylim(1,100)
else:
    plt.semilogy(multiples, E_inv_conds, 'k.-')
    plt.semilogy([penultimate_omega]*2,[min(E_inv_conds),max(E_inv_conds)],'y--')
    plt.xlabel('$\omega_K/\omega_1$', fontsize=14)
    plt.title('$outliers \,\, at: \,\, '+str(E_inv_peaks)[1:-1]+'$', fontsize=6)
plt.ylabel('$\kappa(F^{-1})$', fontsize=14)
#plt.tight_layout()
# save plot and close
print('\n\t'+'saving final image...', end='')
file_name = plot_name+'.png'
plt.savefig(file_name, dpi=300)
print('figure saved: '+plot_name)
plt.close(plot_name)
# open the saved image, if desired
if auto_open:
    webbrowser.open(file_name)
    
# plot the df_HB percent-error result
plot_name = 'df_HB_error_vs_freq_ratio'
auto_open = False
plt.figure(plot_name)
if plot_nondim_omegas:
    plt.semilogy(nondim_omegas, average_errors, 'k.-')
    plt.semilogy([nondim_penultimate_omega]*2,[min(average_errors),max(average_errors)],'y--')
    plt.xlabel('$\delta^*_{\omega_K}=2\,\\frac{\omega_K-\omega_1}{\omega_K+\omega_1}$', fontsize=14)
    plt.title('$outliers \,\, at: \,\, '+str(df_error_nondim_peaks)[1:-1]+'$', fontsize=7)
    plt.semilogy(nondim_omegas,[df_percent_tol]*len(nondim_omegas),'r--')
else:
    plt.semilogy(multiples, average_errors, 'k.-')
    plt.semilogy([penultimate_omega]*2,[min(average_errors),max(average_errors)],'y--')
    plt.xlabel('$\omega_K/\omega_1$', fontsize=14)
    plt.title('$outliers \,\, at: \,\, '+str(df_error_peaks)[1:-1]+'$', fontsize=7)
    plt.semilogy(multiples,[df_percent_tol]*len(multiples),'r--')
plt.ylabel('$\% \,\, error \,\, in \,\, f_{,t_{HB}}$', fontsize=14)
plt.tight_layout()
# save plot and close
print('\n\t'+'saving final image...', end='')
file_name = plot_name+'.png'
plt.savefig(file_name, dpi=300)
print('figure saved: '+plot_name)
plt.close(plot_name)
# open the saved image, if desired
if auto_open:
    webbrowser.open(file_name)
##########################################
# [time accurate] explicit forward euler #
##########################################
delta_t = 0.05             # time step
initial_value = 2.0        # intitial condition

t_start = 0                # initial time
t_end = 100                 # approx. final time (stop at or just after)

time_points = int(math.ceil((t_end-t_start)/delta_t)+1)

# print message to the screen
print('\ncomputing time-accurate solution...\n')

# time stepping
times = []
f = []
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
n_images = time_points          # total number of images computed
skip_images = 25                # images to skip between animation frames
auto_play = False                # automatically play the movie
auto_open = False                # automatically open the final image

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
plt.title(title+'$\Delta t = \,'+str(delta_t)+'$')
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
        plt.plot(times[:n+1],f[:n+1],'b-')
        # progress monitor
        percent_done = float(n)*100.0/(n_images-1)
        print('\tcapturing fig. '+plot_name+' (frame #'+str(frame)+'): ', \
               round(percent_done,2),'%')
        writer.grab_frame()
        frame += 1
    writer.grab_frame()
# plotting: save an image of the final frame
print('\n\t'+'saving final image...', end='')
file_name = plot_name+'.png'
plt.savefig(file_name, dpi=500)
print('figure saved: '+plot_name+'\n')
# free memory used for the plot
plt.close(fig)
# start playing the movie, if desired
if animate_plot and auto_play:
    webbrowser.open(plot_name+'.mp4')
# open the saved image, if desired
if auto_open:
    webbrowser.open(file_name)

##########################################################################
# Find the solution over the lowest-frequency period using the HB method #
##########################################################################

# constant value of the initial guess for the HB solution
init_guess = 2.0

# pseudo-time step size
delta_tau = 0.0783
#delta_tau = 0.16   # for init guess 10.0
#delta_tau = 0.27    # for init guess 15.0
delta_tau = 0.151
delta_tau = 0.05

# choose a nonlinear iterative scheme
pseudo_transient_continuation = True
picard_iteration = False
scaled_pseudo_transient_continuation = False

# if scaled pseudo-transient continuation is being used, just make sure that 
# regular pseudo-transient continuation is turned on too. also, initiate a 
# starting pseudo-time
if scaled_pseudo_transient_continuation:
    pseudo_transient_continuation = True
    tau = 1.0       # [pseudo seconds]
    
# adjust omega?
adjust_omegas = False
if adjust_omegas:
    actual_omegas = [1.567, 2.34]
    omegas = [1.0, 2.1]
# begin optimizing omega at this convergence criteria
adjust_omega_criteria = 1e-4
# cost "zero" value
C_converged = 1e-16
# perturbation in omega for evaluating finite differences (don't let this be a
# huge decimal. if you do, the "long" period will be very long)
delta_omega = 0.01
# learning rate (step size) for each step in omega
eta = 0.01
# set the time-accurate time step for taking the single Euler steps
step_delta_t = 0.01
# create a list for the frequency evolution history
omegas_history = [copy.copy(omegas)]
# adjusting-omega flag (turns on once residual criteria is met)
now_adjusting_omegas = False

# residual convergence criteria
residual_convergence_criteria = 1e-10

# maximum number of pseudo-time steps to try
max_pseudo_steps = 170000

# print message to the screen
print('computing the harmonic-balance solution...\n')

# create the harmonic balance operator matrix and find the time instances
D_HB, t_HB = harmonic_balance_operator(omegas, time_discretization)
# create a constant-valued initial guess for the HB solution
f_HB = np.array([init_guess]*len(t_HB)).reshape((len(t_HB),1))
# create a list for the solution evolution history
f_HB_history = [np.copy(f_HB)]
# create a list for the residual evolution history
residual_history = []
# create a list for the cost evolution history
C_history = []


# start the pseudo-transient continuation method
for k in range(max_pseudo_steps):
    
    if now_adjusting_omegas:
        # note the time instances and omegas from the previous iteration
        t_HB_old = t_HB
        omegas_old = omegas_history[-1]
        # if we're changing the omegas, then recompute the time instances, the
        # operator matrix, and interpolate the solution onto the new time instances
        D_HB, t_HB = harmonic_balance_operator(omegas, time_discretization)
        t_HB, f_HB, dummy = fourierInterp_given_freqs(t_HB_old, f_HB, omegas_old, t_HB)
    else:
        # for the first adjustment iteration's interpolation step
        omegas_old = omegas
    
    # compute the residual vector corresponding the current solution
    func_evaluations = my_non_periodic_ode(t_HB,f_HB,actual_omegas)
    matrix_vector_product = np.dot(D_HB,f_HB)
    residual = func_evaluations - matrix_vector_product

    # compute the norm of the residual vector and print to the screen
    norm_residual = np.linalg.norm(residual)
    residual_history.append(norm_residual)    
    print('\titer: '+str(k)+'\t||residual||: '+str(norm_residual), end='')

    # compute the "error," which is a function of the residual
    if pseudo_transient_continuation:
        I = np.eye(2*len(omegas)+1)     # identity matrix
        step_size = delta_tau
        print()
        if scaled_pseudo_transient_continuation:
            # q = 1 and v = -1 is just regular pseudo-transient continuation
            v = -1
            q = lambda tau: 1           # differentiable w/ q(0)=1
            # try a new scaling here
            power_on_tau = 1.0
            dovishness = 1          # opposite of "agressiveness" (must be > 0)
            v = -1
            q = lambda tau: 1-(tau/max_pseudo_steps)**dovishness # differentiable w/ q(0)=1
            step_size = -(v*delta_tau)/q(tau)
            print('\tpseudo step: '+str(round(step_size,3)))
            tau += delta_tau
        B = step_size*I
    if picard_iteration:
        B = np.linalg.inv(D_HB)         # inverse of HB operator
    
    error = np.dot(B,residual)
    
    # if convergence criteria is met, end, else, update solution
    if norm_residual < residual_convergence_criteria:
        # converged solution found
        print('\n\t\tharmonic balance solution found.\n')
        break
    elif np.isnan(norm_residual) or np.isinf(norm_residual):
        # unstable solution
        print('\n\t\tunstable solution. try again.\n')
        break
    else:
        # update solution and append to solution history
        f_HB += error
        f_HB_history.append(np.copy(f_HB))
        # if the residual is low enough, update the frequencies
        if adjust_omegas:
            # compute the "long"-period cost for the updated solution
            C = compute_cost(t_HB, f_HB, omegas, step_delta_t, actual_omegas)
            #if C < C_converged:
            #    adjust_omegas = False
            if norm_residual < adjust_omega_criteria:
                # find the gradient of the cost function with respect to each of 
                # the K angular frequencies using finite differencing
                gradient_C_omega = []
                for i in range(len(omegas)):
                    # preturb the i-th angular frequency
                    omegas_perturbed = copy.copy(omegas)
                    omegas_perturbed[i] = omegas[i] + delta_omega
                    # compute the new time instances and operator (unless we're 
                    # perturbing the first omega, the time instances don't change)
                    D_HB_perturbed, t_HB_perturbed = harmonic_balance_operator(omegas_perturbed, time_discretization)
                    # if it's the first omega that's being perturbed, then the time
                    # instances are different from what they used to be. use 
                    # fourier interpolation given the unperturbed omegas to 
                    # interpolate f_HB before the update onto t_HB_perturbed
                    f_HB_old = f_HB_history[-2]
                    if i == 0:
                        t_HB_perturbed, f_HB_old, dummy = fourierInterp_given_freqs(t_HB, f_HB_old, omegas_old, t_HB_perturbed)
                    # evaluate the differtial equation using the perturbed times
                    func_evaluations_perturbed = my_non_periodic_ode(t_HB_perturbed,f_HB_old,actual_omegas)
                    # compute the residual coming from the perturbed quantites
                    f_HB_old = np.array(f_HB_old).reshape(len(f_HB_old),1)
                    matrix_vector_product_perturbed = np.dot(D_HB_perturbed,f_HB_old)
                    residual_perturbed = func_evaluations_perturbed - matrix_vector_product_perturbed
                    # compute the error using the same B matrix used above
                    error_perturbed = np.dot(B,residual_perturbed)
                    # compute the solution update based on the perturbed quantities
                    f_HB_perturbed = f_HB_old + error_perturbed
                    # compute the cost associated with the perturbed quantities
                    C_perturbed = compute_cost(t_HB_perturbed, f_HB_perturbed, omegas_perturbed, step_delta_t, actual_omegas)
                    # compute the partial derivative of the cost w.r.t. omega_i
                    dC_domega_i = (C_perturbed - C)/delta_omega
                    # append this component of the gradient
                    gradient_C_omega.append(dC_domega_i)
                # update the omegas using gradient descent, and satisfy constraints
                new_omega_candidates = [omega-eta*derivative for omega,derivative in zip(omegas,gradient_C_omega)]
                for i in range(len(omegas)):
                    if new_omega_candidates[i] <= 0.0:
                        omegas[i] = omegas[i]
                    else:
                        omegas[i] = new_omega_candidates[i]
                #print('new_omega = ',new_omega_candidates)
                
                #omegas = sorted(omegas)
                 
                #if i == 0:
                #    lower_bound = 0.0
                #    upper_bound = new_omega_candidates[i+1]
                #elif i == len(omegas)-1:
                #    lower_bound = new_omega_candidates[i-1]
                #    upper_bound = 100*new_omega_candidates[i] # something big, basically inf
                #else:
                #    lower_bound = new_omega_candidates[i-1]
                #    upper_bound = new_omega_candidates[i+1]
                #if new_omega_candidates[i] > lower_bound and new_omega_candidates[i] < upper_bound:
                #    omegas[i] = new_omega_candidates[i]
                
            # record the cost for the last set of frequencies
            C_history.append(C)
        # record the old set of omegas
        omegas_history.append(omegas)

                
                
                
                
            
        
                    
                
            
        

# plotting: USER INPUTS! do you want to animate the solution history or just
# plot the final result? (True = animate, False = just print final result)
animate_plot = False
plot_name = 'harmonic-balance ODE'
n_images = k+1      # total number of images computed
skip_images = 3     # images to skip between animation frames
auto_play = False   # automatically play the movie
auto_open = False   # automatically open the final image

# plotting: instantiate the figure
fig = plt.figure(plot_name)
# plotting: rescale the figure window to fit both subplots
xdim, ydim = plt.gcf().get_size_inches()
# for two plots, this scaling can't be more than 1.7!!!
plt.gcf().set_size_inches(1.8*xdim, 1.8*ydim, forward=True)
# set the title for the HB solution plot
title = ''
counter=1
for omega in omegas:
    title = title + '$\omega_{'+str(counter)+'} ='+str(omega)+'\quad $'
    counter += 1
# things that won't change for the residual history plot
plt.subplot(2,2,2)
plt.xlabel('$iteration$', fontsize=16)
plt.ylabel('$\\left\Vert \\frac{\partial f}{\partial t} \minus D_{HB}f_{HB} \\right\Vert_2$', fontsize=16)
plt.title(r'$\Delta\tau = '+str(delta_tau)+'$')
plt.xlim(0, k)
min_power = int(math.log(min(residual_history),10))-1
max_power = int(math.log(max(residual_history),10))+1
plt.ylim(pow(10,min_power), pow(10,max_power))
# things that won't be changing for the omega-history plot
plt.subplot(2,2,3)
plt.xlabel('$angular \,\,frequency \,\, \#$', fontsize=16)
plt.ylabel('$\omega_i$', fontsize=16)
# things that won't change for the cost history plot
plt.subplot(2,2,4)
plt.xlabel('$iteration$', fontsize=16)
plt.ylabel('$C$', fontsize=16)

# plotting: set the total number of frames
if animate_plot == True:
    # capture all frames (skipping, if necessary) and the final frame
    all_frames = list(range(0,n_images,skip_images+1))+[n_images-1]
else:
    # no animation: just capture the last one
    all_frames = [n_images-1]
# plotting: capturing the movie
writer = animation.writers['ffmpeg'](fps=10)
with writer.saving(fig, plot_name+'.mp4', 300):
    frame = 0
    for n in all_frames:
        # plot the HB solution
        plt.subplot(2,2,1)
        plt.cla()
        plt.plot(t_HB,f_HB_history[n],'mo')
        t_HB_int, f_HB_int, dummy = fourierInterp_given_freqs(t_HB, f_HB_history[n], omegas)
        plt.plot(t_HB_int, f_HB_int, 'm--')
        plt.xlabel('$t$', fontsize=16)
        plt.ylabel('$f_{HB}$', fontsize=16)
        plt.ylim(np.min(f_HB_history), np.max(f_HB_history))
        plt.title(title)
        # plot the residual
        plt.subplot(2,2,2)
        if n > 0 and residual_history[n] >= residual_history[0]:
            plt.semilogy(residual_history[:n+1],'g-')
        else:
            plt.semilogy(residual_history[:n+1],'r-')
        # plot the omegas
        plt.subplot(2,2,3)
        for j in range(n):
            if j == 0:
                pattern = 'go'
            elif j == n-1:
                pattern = 'ro'
            else:
                pattern = 'k.'
            plt.plot(list(range(1,len(omegas)+1)),omegas_history[j],pattern)
        plt.xlim(0,len(omegas)+1)
        # plot the cost
        plt.subplot(2,2,4)
        if residual_history[n] >= adjust_omega_criteria:
            pattern = 'b.'
        else:
            pattern = 'm.'
        plt.semilogy(C_history[:n+1],pattern)
        # set the spacing options
        plt.tight_layout()
        #plt.subplots_adjust(left=0.07)
        #plt.subplots_adjust(right=0.95)
        # progress monitor
        percent_done = float(n)*100.0/(n_images-1)
        print('\tcapturing fig. '+plot_name+' (frame #'+str(frame)+'): ', \
               round(percent_done,2),'%')
        writer.grab_frame()
        frame += 1
    writer.grab_frame()
# rescale the y-axis of the HB solution plot before saving an image
plt.subplot(2,2,1)
white_space = (max(f_HB_int)-min(f_HB_int))/5.0
plt.ylim(min(f_HB_int)-white_space,max(f_HB_int)+white_space)
# plotting: save an image of the final frame
print('\n\t'+'saving final image...', end='')
file_name = plot_name+'.png'
plt.savefig(file_name, dpi=500)
print('figure saved: '+plot_name+'\n')
# free memory used for the plot
plt.close(fig)
# start playing the movie, if desired
if animate_plot and auto_play:
    webbrowser.open(plot_name+'.mp4')
# open the saved image, if desired
if auto_open:
    webbrowser.open(file_name)

##########################################################
# compare the harmonic-balance and time-accurate results #
##########################################################

# find the average value of the harmonic-balance solution
f_ave_HB = np.average(f_HB_int)

# starting from the end of the time-accurate simulation, figure out how many
# time-accurate time steps are required to just traverse the "long" period of 
# the actual solution. Interpolate that number of points of the time-accurate 
# solution, starting from the end of the simulation onto a grid that has one
# less point than the one being interpolated. the average of the time-accurate
# solution is taken to be the average value of this interpolation.
print('interpolating the TA solution to onto exactly one "long" period (for averaging)...\n')
points_long_period = math.ceil(T_actual_sol/delta_t)+1
t_last_long_period = myLinspace(t_end-T_actual_sol,t_end,points_long_period-1)
t_last_long_period, f_last_long_period = linearInterp(times[-points_long_period:],f[-points_long_period:],t_last_long_period,verbose=True)
f_ave_TA = sum(f_last_long_period)/(points_long_period-1)

# interpolate the HB solution onto a grid that has same interval as TA solution
print('\ninterpolating the HB solution onto a grid with the same interval as the TA solution...\n')
T_lowest_omega = 2.0*np.pi/min(omegas)
N_delta_t_per_T_HB = math.floor(T_lowest_omega/delta_t)
T_HB_check = N_delta_t_per_T_HB*delta_t
N_checkpoints = N_delta_t_per_T_HB+1
t_HB_check = myLinspace(0.0, T_HB_check, N_checkpoints)
t_HB_check,f_HB_check = linearInterp(t_HB_int, f_HB_int, t_HB_check, verbose=True)

# figure out where along the "long" period of the time-accurate solution the 
# harmonic-balance solution lies
min_norm_diff = 1e6
norm_diffs = []
min_index = 0
t_HB_int_min_shifted = []       # interpolated HB sol at the best minimum
t_HB_min_shifted = []           # HB sol at the time instances at best minimum
t_HB_int_trial_shifted = []     # "proving" (sliding) interpolated HB solution
for i in range(-time_points,-N_checkpoints):
    current_f_TA_range = f[i:i+N_checkpoints]
    t_HB_int_current = [t+times[i] for t in t_HB_check]
    diff = [abs(f_HB_check[j]-current_f_TA_range[j]) for j in range(N_checkpoints)]
    norm_diff = myNorm(diff)
    if norm_diff < min_norm_diff:
        min_norm_diff = norm_diff
        min_index = i
        t_HB_int_min_current = t_HB_int_current
        t_HB_min_current = [t+times[i] for t in t_HB]
    norm_diffs.append(norm_diff)
    t_HB_int_min_shifted.append(t_HB_int_min_current)
    t_HB_min_shifted.append(t_HB_min_current)
    t_HB_int_trial_shifted.append(t_HB_int_current)
print('\ngenerating plot/animation of comparison process...\n')

# plotting: USER INPUTS! do you want to animate the solution history or just
# plot the final result? (True = animate, False = just print final result)
animate_plot = False
plot_name = 'sliding_comparison_HB_with_TA'
n_images = len( t_HB_min_shifted)        # total number of images computed
skip_images = 10                # images to skip between animation frames
auto_play = False                # automatically play the movie
auto_open = False                # automatically open the final image

# plotting: initializations
fig = plt.figure()
# plotting: things that will not be changing inside the loop
vertical_padding = (max(f)-min(f))/4.0
title = ''
counter=1
for omega in omegas:
    title = title + '$\omega_{'+str(counter)+'} ='+str(omega)+'\quad $'
    counter += 1

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
        plt.cla()
        plt.plot(times,f,'b-')
        plt.plot(t_HB_min_shifted[n],f_HB,'go')
        plt.plot(t_HB_int_min_shifted[n],f_HB_check,'g-')
        if not n==n_images-1:
            plt.plot(t_HB_int_trial_shifted[n],f_HB_check,'m-')
        plt.title(title+'$\Delta t = '+str(delta_t)+'\quad min \, \\left\Vert f \minus f_{HB} \\right\Vert_2 = '+str(round(min(norm_diffs[:n+1]),2))+'$')
        plt.xlabel('$t$', fontsize=18)
        plt.ylabel('$f(t)$', fontsize=18)
        plt.xlim(0,t_end)
        plt.ylim(min(f)-vertical_padding,max(f)+vertical_padding)
        # progress monitor
        percent_done = float(n)*100.0/(n_images-1)
        print('capturing fig. '+plot_name+' (frame #'+str(frame)+'): ', \
               round(percent_done,2),'%')
        writer.grab_frame()
        frame += 1
    writer.grab_frame()
# plotting: save an image of the final frame
print('\n\t'+'saving final image...', end='')
file_name = plot_name+'.png'
plt.savefig(file_name, dpi=500)
print('figure saved: '+plot_name+'\n')
# free memory used for the plot
plt.close(fig)
# start playing the movie, if desired
if animate_plot and auto_play:
    webbrowser.open(plot_name+'.mp4')
# open the saved image, if desired
if auto_open:
    webbrowser.open(file_name)

################################################
# print all intermediate results to the screen #
################################################

# actual periods of the HB solution and the actual ODE solution
print('\n-true periods of solutions, based on given angular frequencies:')
print('\n\tperiod of HB solution:', round(T_HB_sol,3))
print('\n\tperiod of the ODE solution:', round(T_actual_sol,3),'\n')

# print information about the HB operator
print('\n-a look at the HB operator:\n')
print('\tomegas =', omegas)
print('\n\tK = '+str(len(omegas)))
print('\n\tN = '+str(2*len(omegas)+1))
print('\n\tD_HB = ', str(np.around(D_HB,3))[1:-1].replace('\n','\n\t'+' '*7),'\n')
print('\tdet(D_HB) =', np.linalg.det(D_HB),'\n')
print('\tcond(D_HB) =', np.linalg.cond(D_HB),'\n')
#E_inv = HB_forward_transform_matrix(omegas, time_discretization)
#print('\n\tE_inv = ', str(np.around(E_inv,3))[1:-1].replace('\n','\n\t'+' '*7),'\n')

# print both TA and HB average function values to the screen
print('\n-comparing the average value of the time-accurate solution (over \n'
       +' one steady-state "long" period corresponding to the actual \n'
       +' frequencies) to the average value of the harmonic-balance solution \n'
       +' (over the '+str(2*len(omegas)+1)+' time instances):')
print('\n\tf_ave (time-accurate) = '+str(round(f_ave_TA,3)))
print('\tf_ave (harmonic-balance) = '+str(round(f_ave_HB,3))+'\n')

