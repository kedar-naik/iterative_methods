# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 01:16:35 2016
@author: Kedar
"""

import math
from matplotlib import pyplot as plt
from matplotlib import animation         # for specifying the writer
import numpy as np
import webbrowser
import copy
from functools import reduce
import sys                               # for finding machine zero

from time_spectral import myLinspace, myNorm, linearInterp
from iterative_methods import my_inv, my_pinv, my_rank
from dft_practice import my_dft


#-----------------------------------------------------------------------------#
def period_given_freqs(omegas):
    '''
    if the frequencies that make up your solution are all rational numbers, the
    resulting signal is a periodic function. given that set of frequencies, 
    this function will return the corresponding period.
    '''
    import decimal
    from functools import reduce
    from math import gcd, pi
    # find the maximum number of decimal points seen the given frequencies
    decimal_pts = [-decimal.Decimal(str(f)).as_tuple().exponent for f in omegas]
    max_dec_pts = max(decimal_pts)
    # multiply all frequencies by the power of ten that make them integers
    scaled_omegas = [int(omega*pow(10,max_dec_pts)) for omega in omegas]
    # find the greatest common divisor of these scaled frequencies
    GCD = reduce(lambda x,y: gcd(x,y), scaled_omegas)
    # find the total period
    total_period = 2.0*pi/(GCD/pow(10,max_dec_pts))
    return total_period
#-----------------------------------------------------------------------------#
def my_non_periodic_fun(t, omegas, verbose=False):
    '''
    given a set of specfied frequencies and time points, this subroutine 
    samples a function composed of a sum of sinusoids oscillating at the 
    frequencies specified
    returns numpy column vectors
    '''
    #omegas += [8.6]         # add an additional angular frequency. this ensures
                            # that the HB expansion will never be perfect.
    import numpy as np
    f = []
    df = []
    equation_string = 'f(t) = '
    # run through all the time samples
    counter = 1
    for t_i in t:
        f_i = 0 + 60.4
        f_i = 0 + 60.0
        df_i = 0
        if counter == 1:
            equation_string += str(f_i)
        # run through all the frequencies
        for omega in omegas:
            # f = sum of sines
            #f_i += math.sin(omega*t_i)
            #df_i += omega*math.cos(omega*t_i)

            # f = sum of amplified sines and cosines
            #a_i = -np.sin(omega)/np.sqrt(omega)
            #b_i = np.cos(omega)**3
            a_i = 0.2*omega     # sine coeff
            b_i = omega**2.0                  # cosine coeff
            f_i += a_i*math.sin(omega*t_i) + b_i*math.cos(omega*t_i)
            df_i += omega*(a_i*math.cos(omega*t_i) - b_i*math.sin(omega*t_i))
            if counter == 1:
                equation_string += ' + '+str(round(a_i,3))+'*sin('+str(omega)+'*t) + '+str(round(b_i,3))+'*cos('+str(omega)+'*t)'
            
            # f = sum of squared sines (this is not tonal! HB will fail!)
            #f_i += math.sin(omega*t_i)**2.0
            #df_i += 2.0*math.sin(omega*t_i)*omega*math.cos(omega*t_i)
        f.append(f_i)
        df.append(df_i)
        counter += 1
    # turn f and df into numpy column vectors
    f = np.array(f)
    f = f.reshape(f.size,1)
    df = np.array(df)
    df = df.reshape(df.size,1)
    if verbose:
        print('\nfunction being used:\n\n\t'+equation_string+'\n')
    return f, df
#-----------------------------------------------------------------------------#
class governing_ode:
    '''
    class for defining and evaluating the ode 
    '''
    # class constructor (accepts the actual omegas from the user)
    def __init__(self, actual_omegas):
        self.actual_omegas = actual_omegas
    
    # function to evaluate the expression that follows du/dt = ...
    def evaluate(self, t, u):
        '''
        given a set of specfied frequencies and time points, this subroutine 
        samples a function (meant the represent the RHS of an ODE) composed of 
        a sum of sinusoids oscillating at the frequencies specified. returns a 
        numpy array of the samples of the RHS of the ODE.
        '''
        import numpy as np
        # initialize single-entry flag
        single_entry = False
        # if inputs are floats/ints, convert to lists
        if type(t)==np.int_ or type(t)==np.float_ or type(t)==int or type(t)==float:
            t = [t]
            single_entry = True
        if type(u)==np.int_ or type(u)==np.float_ or type(u)==int or type(u)==float:
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
            for omega in self.actual_omegas:
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
def parse_time_discretization_string(omegas, time_discretization):
    '''
    read in the time-discretization string and parse out the different pieces,
    namely: the discretization style (i.e. whether we're using T1 or Nyquist),
    the type of spacing (i.e. whether the spacing is uniform or arbitrary), and
    the number of time instances.
    '''
    # compute N_0
    K = len(omegas)     # number of angular frequencies provided
    N_0 = 2*K+1         # minimum points required for trig interpolation
    # initialize to defaults (use_Nyquist, uniform spacing (not random), N=N_0)
    options = ['use_Nyquist', False, N_0]    
    # parse the time_discretization string passed in
    options_passed_in = time_discretization.split('-')
    # run through the tokens and set the various options
    for option in options_passed_in:
        if option:
            if option[:4]=='use_':
                options[0] = option
            elif option=='random':
                options[1] = True
            else:
                options[2] = int(option)
    # rename the options for clarity
    discretization_style = options[0]   # (use_T1, use_Nyquist)
    nonuniform_spacing = options[1]     # (True, False)
    N = options[2]                      # an integer
    # return the options found
    return discretization_style, nonuniform_spacing, N
#-----------------------------------------------------------------------------#
def HB_time_instances(omegas, time_discretization):
    '''
    this function returns a numpy array corresponding to the desired time
    discretization:
      - time_discretization = a string that accepts various flags
      -                    -> 'use_T1' (equally spaced points just spanning the
                                  period corresponding to the lowest frequency)
      -                    -> 'use_Nyquist' (use Nyquist criterion for spacing)
      -                    -> '-random' (use a random/nonuniform spacing)
      -                    -> '-5' (# of time points desired, default: 2*K+1)
      - the default is: 
                  time_discretization = 'use_Nyquist'
        which means, "create a set of 2*K+1 uniformly spaced time instances 
        that start at zero and are spaced such that they just satisfy the time-
        interval requirement dictated by the Nyquist rate"
      - but can accept complicated arguments such as:
                  time_discretization = 'use_Nyquist-random-11'
        which means, "randomly select 11 time instances that satisfy the time-
        interval required by the Nyquist rate."
      - or, can also accept an argument like:
                  time_discretization = 'use_T1-random'
        which means "randomly distribute 2*K+1 time instances within [0, T1)"
      - or, another example:
                  time_discretization = 'use_T1-15'
        which means, "create a set of 15 uniformly spaced time instances that
        are just shy of spanning the range [0, T1]"
    '''
    import numpy as np
    # parse the time_discretization string and extract the relevant information
    discretization_style, \
    nonuniform_spacing, \
    N = parse_time_discretization_string(omegas, time_discretization)
    # create the set of time instances as directed
    if discretization_style=='use_T1':
        # time period corresponding to the lowest frequency
        T_lowest_omega = (2.0*np.pi)/min(omegas)
        if nonuniform_spacing:
            # randomly pick N points in the range [0,T1)
            t_HB = sorted(T_lowest_omega*np.random.rand(N))
        else:
            # compute the time interval
            delta_t = T_lowest_omega/N
            # set the location of the time instances
            t_HB = np.array([i*delta_t for i in range(N)])
    if discretization_style=='use_Nyquist':
        # scaling up nyquist frequency by this factor (must be > 1)
        scaling_fac = 1.01      # just satisfies Nyquist
        #scaling_fac = 0.21       # fails to satisfy Nyquist (but will often work, b/c matrix will still be full rank!)
        #scaling_fac = 0.5       # fails to satisfy Nyquist (but will never work because columns are periodic with 1 in the ratio omega_k/omega_K, so omega_0 = omega_-K + omega_K, every possible omega_K is inadmissible and )
        #scaling_fac = 1.5       # should result in bad omegas...or maybe not...looks like not
        # nyquist rate (this is a bandlimited signal)
        nyquist_rate = max(omegas)/np.pi
        # find the corresponding time interval
        delta_t = 1.0/(scaling_fac*nyquist_rate)
        if nonuniform_spacing:
            # set the first time instance, randomly within [0.0, delta_t)
            t_HB = [float(delta_t*np.random.rand())]
            # set the rest of the time instances, maintaining the time interval 
            # corresponding to the nyquist rate
            for i in range(N-1):
                t_HB.append(t_HB[-1] + float(delta_t*np.random.rand()))
            # convert to a numpy array
            t_HB = np.array(t_HB)
        else:
            # set the location of the time instances
            t_HB = np.array([i*delta_t for i in range(N)])
    return t_HB
#-----------------------------------------------------------------------------#
def HB_expansion_matrix(omegas, time_discretization):
    '''
    given the discrete set of nonzero angular frequencies and the time
    instances of interest, this subroutine returns the G matrix that 
    represents the forward discrete trig expansion taken at the specific 
    frequencies. if the time points to use are already known, then pass them in
    under time_discretization. if they're not known, then provide the typical
    string needed by the HB_time_instances subroutine.
    '''
    import numpy as np
    # if not provided, set the location of the time instances
    if type(time_discretization)==str:
        t_HB = HB_time_instances(omegas, time_discretization)
    else:
        t_HB = time_discretization
    # extact the number of time instances
    N = len(t_HB)
    # compute the value of N_0
    K = len(omegas)
    N_0 = 2*K+1           # required for finding the trig coefficients
    '''
    # create a list of all N discrete frequencies
    # w = [0, w_1, ..., w_K, -w_K, ..., -w_1]
    w = [0]+omegas+[-omegas[-(i+1)] for i in range(K)]
    # create the forward-transform matrix, E_inv
    E_inv = np.zeros([N,N], dtype=np.complex_)
    for i in range(N):
        for j in range(N):
            E_inv[i][j] = (1/N)*np.exp(1j*w[j]*t_HB[i])
    return E_inv
    '''
    # create a list of all 2*K+1 discrete frequencies, from -K to +K
    # w = [-w_K, ..., -w_1, 0, w_1, ..., w_K]
    w = [-omegas[-(i+1)] for i in range(len(omegas))]+[0.0]+omegas
    # create the forward-transform matrix, G
    G = np.zeros([N,N_0], dtype=np.complex_)
    for i in range(N):
        for j in range(N_0):
            G[i][j] = np.exp(1j*w[j]*t_HB[i])
    return G
#-----------------------------------------------------------------------------#
def harmonic_balance_operator(omegas, time_discretization='use_Nyquist'):
    '''
    given a discrete set of frequencies and a time discretization (either 
    instructions on how to select the time instances in a string OR the actual
    time instances to use as a numpy array) compute the harmonic-balance 
    differential operator matrix
    '''
    import numpy as np
    
    # if not provided, set the location of the time instances
    if type(time_discretization)==str:
        # set the location of the time instances, as directed
        t_HB = HB_time_instances(omegas, time_discretization)
    else:
        t_HB = time_discretization
    # create a list of all 2*K+1 discrete frequencies, from -K to +K
    # w = [-w_K, ..., -w_1, 0, w_1, ..., w_K]
    w = [-omegas[-(i+1)] for i in range(len(omegas))]+[0.0]+omegas
    # create the diagonal matrix holding the frequencies
    w_imag = [1j*w_i for w_i in w]
    Omega = np.diag(w_imag)
    # construct the forward-tranform matrix
    G = HB_expansion_matrix(omegas, time_discretization=t_HB)
    # check to see if G has full rank:
    # n.b. "full rank" means either full column rank OR full row rank. in other
    # words, if a matrix has full rank, it has linearly independent columns,
    # linearly independent rows, or both. for an m-by-n matrix, check for full
    # rank by seeing if r = min(m,n)
    r = my_rank(G)
    print('rank(G) = ', r)
    print('np.linalg.matrix_rank(G) =', np.linalg.matrix_rank(G))
    shorter_dimension = min(np.shape(G))
    if r == shorter_dimension:
        full_rank = True
    else:
        full_rank = False
    # take the (approximate) pseudoinverse of G and construct the operator
    if full_rank:
        # compute the pseudoinverse of the forward-transform matrix (n.b. if G
        # is square and has full rank, then it's invertible. the pseudoinverse
        # will be identical to the inverse of G)
        G_plus = my_pinv(G)
        #G_plus = np.linalg.pinv(G) #HACK
        # compute the operator (D_HB = G*Omega*G_plus)
        # it's always real...proven.
        D_HB = np.real(reduce(lambda x,y: np.dot(x,y), [G, Omega, G_plus]))
    else:        
        # compute an approximate pseudoinverse of the forward-tranform matrix
        G_tilde_plus = my_pinv(G, approximate_using_svd=True)
        # compute the operator (D_HB = G*Omega*G_tilde_plus)
        # it's always real... proven.
        D_HB = np.real(reduce(lambda x,y: np.dot(x,y), [G, Omega, G_tilde_plus]))
    # return the operator matrix and the time instances
    return (D_HB, t_HB)
#-----------------------------------------------------------------------------#
def trig_interp_given_freqs(x, y, omegas, x_int=False, return_coeffs=False):
    '''
    
    (This function should probably be called 'trig_interp' because it's not a
     Fourier series if you're just picking a handful of angular frequencies.)
    This function interpolates a given set of ordinates and abscissas with a
    trigonometric series that uses only a specific set of frequencies (tones). 
    The interpolation is constructed using coefficients found for the 
    corresponding complex-exponential terms by solving a linear system built up 
    from the given abscissas and ordinates.
    if the number of samples is not equal to 2K+1, where K=len(omegas), then
    the linear system can only be solved in a least-squares sense. that is, 
    instead of using the inverse of G, one must take the pseudoinverse of G.
    once the coefficients are found, they are projected onto a fine 
    (interpolant) grid by multiplication with a G_int matrix, which is a matrix
    of complex exponential samples, just like G, but with a row for each point
    in x_int.
    Input:
      - abscissas, x (as a list) (leave out last, duplicate point in period)
      - ordinates, y (as a list) (again, leave out last point, if periodic)
      - angular frequencies or "tones", omegas (as a list)
      - new abscissas, x_int (as a list) (optional! defaults to 10x refinement)
    Output:
      - new abscissas, x_int (as a list)
      - interpolated ordinates, y_int (as a list)
      - (derivative of the interpolant, dydx_int (as a list))
    '''
    import numpy as np
    # refinment factor for the interpolant. (If there are originally 10 points
    # but you want the Trigonometric Series to be defined on 50 points, then  
    # the refinement factor is 5.)
    refine_fac = 10
    # number of points passed in
    n = len(x)
    # rename and reshape the ordinates into a column vector
    y_vec = np.array(y).reshape(n,1)
    # if zero has not been included as a frequency, add it
    #if 0 not in omegas:
    #    omegas = [0.0] + omegas
    # set x_int, if it hasn't been given
    if type(x_int) == bool:
        n_int = refine_fac*(n)
        x_int = myLinspace(x[0],x[-1]+x[1],n_int)
    else:
        n_int = len(x_int)
        
    # compute the harmonic-balance expansion matrix for the given points
    G = HB_expansion_matrix(omegas, time_discretization=x)
    
    # compute HB expansion coefficients by taking the pseudoinverse of G
    try:
        c_pinv = np.dot(my_pinv(G), y_vec)
    except ValueError:
        print('\n\tDuring interpolation: G matrix doesn\'t have full rank...'+\
              ' using an approximate pseudoinverse...\n')
        c_pinv = np.dot(my_pinv(G, approximate_using_svd=True), y_vec)
    # create a new G matrix based on the time grid we are interpolating onto
    G_fine = HB_expansion_matrix(omegas, time_discretization=x_int)
    # multiply G_fine by the coefficients to construct the interpolant
    y_int = np.real(np.dot(G_fine,c_pinv))
    
    # for the derivative, we're basically just multiplying by the HB operator,
    # except now the HB operator has n_int rows instead of just 2K+1. warning:
    # if you want the derivative to be correct here, you need to make sure the 
    # original abscissas, x, satisfy the Nyquist criterion!
    # anyway, the first step is to create the Omega matrix:
    # create a list of all 2*K+1 discrete frequencies, from -K to +K
    # w = [-w_K, ..., -w_1, 0, w_1, ..., w_K]
    w = [-omegas[-(i+1)] for i in range(len(omegas))]+[0.0]+omegas
    # create the diagonal matrix holding the frequencies
    w_imag = [1j*w_i for w_i in w]
    Omega = np.diag(w_imag)
    # then, we get the derivative dy/dx at the interpolation points by doing
    # G_fine*Omega*pinv(G)*y_vec = G_fine*Omega*c_pinv
    dydx_int = np.real(reduce(lambda x,y: np.dot(x,y), [G_fine, Omega, c_pinv]))
    
    # return the interpolant and the derivative
    return (x_int, y_int, dydx_int)
    
    
    '''
    import math
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
    if return_coeffs:
        return (x_int, y_int, dydx_int, a, b)
    else:
        return (x_int, y_int, dydx_int)
    '''
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
    plt.title('$eig\\left(D_{HB}\\right) \, = \,'+title+'$', y=1.03, fontsize=8)
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
def HB_omega_check(omegas_given, N_time_instances='2K+1', make_plots=True):
    '''
    only use this subroutine if the HB time instances are uniformly spaced and
    just shy of spanning T1 (the period corresponding to the lowest frequency).
    
    this subroutine reads in a list of K angular frequencies and tells the user
    whether the values given will lead to an ill-conditioned HB operator, with
    less than full rank, when the N time instances are uniformly spaced across
    the period corresponding to the lowest angular frequency.
    
    if inadmissible frequencies are found, the user will be given a range (or 
    ranges) of feasible values that could be substituted in its place.

    the function returns a bool, which is True if all given omegas are valid.
    '''
    import math
    from matplotlib import pyplot as plt
    # initialize the boolean that will be returned
    all_omegas_valid = True
    # total number of given angular frequencies
    K = len(omegas_given)
    # compute 2*K+1 
    N_0 = 2*K+1    
    # find the number of time instances being used (no. of rows in G)
    if N_time_instances == '2K+1':
        N = N_0
    else:
        N = N_time_instances
    # set the best-possible rank of matrix G for the given situation
    if N < N_0:
        # for the case where N < N_0 (G has fewer rows than columns -- fewer 
        # time instances than the 2K+1 omega values), the rank of the matrix G
        # can be no greater than the number of rows, N (since row rank = column 
        # rank). BUT, any rank degeneracies found below must still be 
        # subtracted from N_0. if that rank is less than N, then we have
        # a matrix with a rank smaller than the best possible, otherwise, the
        # limiting factor is the row rank, N. so, the rank of the matrix will
        # finally be given by rank(G) = min(N, N_0-degeneracies)
        # n.b. full column rank -- which is what we need to construct the left 
        # pseudoinverse of G -- is impossible when N < N_0. expect poor results
        best_possible_rank = N
        # print warnings to the screen
        print()
        print('\t\t***    WARNING: INSUFFICIENT NUMBER OF TIME INSTANCES   ***')
        if N_0-N == 1:
            print('\t\t***     need to use at least '+str(N_0-N)+' more time instance!     ***')
        else:
            print('\t\t***     need to use at least '+str(N_0-N)+' more time instances!     ***')
        print('\t\t***   otherwise full column rank will not be possible!  ***')
        print('\t\t***      nevermind the omega values until then!         ***')
    else:
        # for the case where N >= N_0, full column rank corresponds to N_0. 
        # even though there may be more than N_0 rows, the rank of G can be, at 
        # most, N_0 (since row rank = column rank). any rank degeneracies found 
        # below must therefore be subtracted from N_0, not N. since full column 
        # rank can be achieved, a left pseudoinverse of G can be constructed 
        # (assuming no causes of degeneracy are found below)
        best_possible_rank = N_0
    # for notational simplicity later, note the first omega
    omega_1 = omegas_given[0]
    # initialize an empty list of to hold dictionaries for inadmissible frequency values
    omega_dicts = []
    for i in range(K):          # i = 0,1,...,K-1
        # current omega
        omega_i = omegas_given[i]
        # range constraints: the i-th frequency must be strictly less than the 
        # one that comes after it and strictly greater than the one that comes 
        # before. the first frequency must be strictly greater than zero. 
        # Likewise, the upper bound for the last frequency is infinity.
        if i == 0:
            lower_bound = 0.0
        else:
            lower_bound = omegas_given[i-1]
        if i == K-1:
            upper_bound = math.inf
        else:
            upper_bound = omegas_given[i+1]
        # find the inadmissible ratios due to Cases 1 and 2 
        inadmissible_values_case1 = []  # all bad values for Case 1
        inadmissible_values_case2 = []  # all bad values for Case 2
        inadmissible_points = []        # feasibly bad values (within bounds)
        case1_max = 0.0
        case2_max = 0.0
        n_multiples = 4             # for Case 1
        no_of_Ns = 3                # for Case 2
        if i > 0:       # omega_1 is never inadmissible under Case 1
            # find the inadmissible ratios due to Case 1 (when the i-th column
            # ends up being real and, therefore, equals its complex conjugate)
            # run through the first no_of_Ns periods of N
            for m in range(1,n_multiples+1):
                # compute the inadmissible ratio for this multiple
                #bad_ratio = m*(K+0.5)   # (only applies if N=N_0)
                bad_ratio = m*N/2.0     # (applied for any N)
                # number of linearly dependent columns of this multiple
                if m%2 == 0:            # even m
                    rank_degeneracy = 2
                    bad_case = 'I.B'    # case I degeneracy w/ even m
                    bad_columns = str(-(i+1))+',0,'+str(i+1)  # record the columns affected
                else:                   # odd m
                    rank_degeneracy = 1
                    bad_case = 'I.A'    # case I degeneracy w/ odd m
                    bad_columns = str(-(i+1))+','+str(i+1)  # record the columns affected
                # compute the resulting inadmissible rank of the matrix
                if N < N_0:
                    bad_rank = min(best_possible_rank, N_0-rank_degeneracy)
                else:
                    bad_rank = best_possible_rank-rank_degeneracy
                # compute the corresponding inadmissible value of this omega
                bad_value = bad_ratio*omega_1
                # if the inadmissible value occurs within the bounds, record it
                # as an inadmissible point (bad_value, bad_ratio, bad_rank, bad_case, bad_columns)
                if bad_value > lower_bound and bad_value < upper_bound:
                    inadmissible_points.append((bad_value, bad_ratio, bad_rank, bad_case, bad_columns))
                # record the (inadmissible value, the corresponding bad rank)
                inadmissible_values_case1.append(bad_value)
            # find the greatest inadmissible value due to Case 1
            case1_max = max(inadmissible_values_case1)
            
        # find the inadmissible ratios and values for Case 2 (two arbitrary 
        # columns in the matrix end up being equal due to N periodicty) note
        # that any omega can be inadmissible under Case 2, including omega_1
        test_omegas = list(omegas_given)
        # remove the current omega from the list and mirror with negation
        test_omegas.pop(i)
        test_omegas = [-test_omegas[-i] for i in range(1,K)] + test_omegas
        # run through the test omegas
        for test_omega in test_omegas:
            # recover the omega no. (i.e. the column no.) correponding to 
            # this test omega
            test_omega_no = int(np.sign(test_omega)*(omegas_given.index(abs(test_omega))+1))
            bad_columns = str(i+1)+','+ str(test_omega_no)
            for r in range(-no_of_Ns,no_of_Ns):
                # compute the inadmissible ratio for this combination
                bad_ratio = test_omega/omega_1 + r*N
                # compute the corresponding inadmissible value
                bad_value = bad_ratio*omega_1
                # note that it's a case 2 degeneracy and also record which 
                # columns interact
                bad_case = 'II'
                # for Case two the rank degeneracy is always 2
                rank_degeneracy = 2
                # the corresponding inadmissible rank is computed as above
                if N < N_0:
                    bad_rank = min(best_possible_rank, N_0-rank_degeneracy)
                else:
                    bad_rank = best_possible_rank-rank_degeneracy
                # if the inadmissible value occurs within the bounds, record it
                # as an inadmissible point (bad_value, bad_ratio, bad_rank, bad_case)
                if bad_value > lower_bound and bad_value < upper_bound:
                    inadmissible_points.append((bad_value, bad_ratio, bad_rank, bad_case, bad_columns))
                # record the (inadmissible value, the corresponding bad rank)
                inadmissible_values_case2.append(bad_value)
        # find the greatest inadmissible value due to Case 2
        case2_max = max(inadmissible_values_case2)
            
        # for the plotting of the K-th omega, set a "upper bound" that 
        # corresponds the minimum inadmissible value from both cases. (for
        # omega_1, case1 doesn't exist, so just use the case 2 max here)
        if i==0:
            K_upper_bound = case2_max
        else:
            K_upper_bound = min(case1_max, case2_max)
        # for the final omega, discard any inadmissible points which exceed the
        # "upper bound" for K
        inadmissible_points = [point_tuple for point_tuple in inadmissible_points if point_tuple[0] <= K_upper_bound]
        # sort the tuples created for inadmissible points by the actual values
        inadmissible_points = sorted(inadmissible_points, key=lambda point_tuple: point_tuple[0])
        # check if the current omega value is at an inadmissible point
        inadmissible_values = [point_tuple[0] for point_tuple in inadmissible_points]
        omega_validity = True
        for inadmissible_value in inadmissible_values:
            if abs(inadmissible_value-omega_i) < 1e-6:
                omega_validity = False
        # create a dictionary for this omega and fill in the known values
        omega_dict = {'value': omega_i,
                      'valid': omega_validity,
                      'lower bound': lower_bound,
                      'upper bound': upper_bound,
                      'K upper bound': K_upper_bound,
                      'inadmissible points': inadmissible_points}
        # append this dictionary to the list of omega dictionaries
        omega_dicts.append(omega_dict)    
    # if there are invalid omegas, print their dictionaries to the screen
    omega_counter = 1
    invalid_omega_counter = 0
    for omega_dict in omega_dicts:
        if not omega_dict['valid']:
            # set the boolean to return False and increment the invalid counter
            all_omegas_valid = False
            invalid_omega_counter += 1
            # print a header
            print('\n\t- OMEGA #'+str(omega_counter)+ ' (' + \
                  str(omega_dict['value']) + ' rad/sec)'+' IS INVALID:')
            # print out the dictionary in a legible manner
            for key in omega_dict:
                if key == 'inadmissible points':
                    print('\t      - full column rank: '+str(N_0))
                    print('\t      - best possible rank: '+str(best_possible_rank))
                    print('\t      - '+(key+': ').ljust(15), end='')
                    row_names = ['value:  ', 'ratio:  ', 'rank(G):', 'case:', 'columns:']
                    for i in range(len(row_names)):
                        print('\n\t\t    '+row_names[i]+'\t', end='')
                        for point in omega_dict[key]:
                            if not type(point[i]) == str:
                                print(str(round(point[i],3)).center(10), end='')
                            else:
                                print(point[i].center(10), end='')
                    print()
                else:
                    print('\t      - '+(key+': ').ljust(15), end='')
                    if not type(omega_dict[key]) == bool:
                        print(str(round(omega_dict[key],3)))
                    else:
                        print(str(omega_dict[key]))
        # increment the counter
        omega_counter += 1
    # print a final warning
    if not all_omegas_valid:
        if invalid_omega_counter == 1:
            print('\n\t\t\t*** there is an inadmissible angular frequency! ***')
        else:
            print('\n\t\t\t*** there are inadmissible angular frequencies! ***')
        if N >= N_0:
            print('\t\t\t***   (try using Nyquist-based discretization)  ***')
    # if desired, for each of the given angular frequencies, plot the  
    # inadmissible values and the corresponding degenerate rank
    if make_plots:
        for i in range(K):
            # extract the inadmissible points for this angular frequency
            bad_points = [point_tuple[0] for point_tuple in omega_dicts[i]['inadmissible points']]
            # extract the corresponding ranks
            bad_ranks = [point_tuple[2] for point_tuple in omega_dicts[i]['inadmissible points']]
            # initialize the boolean
            inadmissible_value = False
            # check to see if the current value is inadmissible (this is a boolean)
            if bad_points:
                index = 0
                for point in bad_points:
                    if abs(omega_dicts[i]['value']-point) < 1e-6:
                        inadmissible_value = True
                        bad_rank = bad_ranks[index]
                    index += 1
            # extract upper and lower bounds and compute the range
            lower_bound = omega_dicts[i]['lower bound']
            upper_bound = omega_dicts[i]['upper bound']
            if upper_bound == math.inf:
                upper_bound = omega_dicts[i]['K upper bound']
            admissible_range = upper_bound-lower_bound
            # full rank is N
            full_rank = N
            # plotting preliminaries
            plot_name = 'omega #'+str(i+1)+' - inadmissible_values'
            auto_open = False
            plt.figure(plot_name)
            # if there are bad points, plot them as vertical lines
            if bad_points:
                plt.plot(bad_points, bad_ranks, 'ko', label='$inadmissible$')
                plt.vlines(bad_points, [0.0]*len(bad_ranks), bad_ranks, 'k')
            # plot a dashed green denoting full rank
            plt.plot([lower_bound, upper_bound],[full_rank]*2,'g--',label='$full \,\, rank$')
            # plot a vertical dashed line denoting the lower bound        
            plt.vlines(lower_bound, 0.0, 1.25*N, 'm', '--', label='$bounds$')
            # if not the last omega, plot a vertical dashed line at the upper bound
            if i != K-1:
                plt.vlines(upper_bound, 0.0, 1.25*N, 'm', '--')
            # plot the given value with the corresponding rank
            if inadmissible_value:
                # if the given value yields less than full rank
                plt.plot(omega_dicts[i]['value'], bad_rank, 'ro', label='$given \,\, value$')
                plt.vlines(omega_dicts[i]['value'], 0.0, bad_rank, 'r','--')
            else:
                # if the given value is fine
                plt.plot(omega_dicts[i]['value'], full_rank, 'bo', label='$given \,\, value$')
                plt.vlines(omega_dicts[i]['value'], 0.0, full_rank, 'b','--')
            # plot labels, axes, title, legend
            plt.xlabel('$\omega_{'+str(i+1)+'}$', fontsize=16)
            plt.ylabel('$rank(G)$', fontsize=16)
            if bad_points:
                # create a string with the inadmissible values
                bad_points_string = ''
                for point in bad_points:
                    bad_points_string += str(round(point,3))
                    if point != bad_points[-1]:
                        bad_points_string += ', '
                # plot the title with inadmissible points
                plt.title('$\omega_{'+str(i+1)+'} ='+str(omega_dicts[i]['value'])+'\quad inadmissible \,\,points:\,\,'+bad_points_string+'$')
            else:
                # plot the title when there are no inadmissible points
                plt.title('$\omega_{'+str(i+1)+'} ='+str(omega_dicts[i]['value'])+'\quad (no \,\, inadmissible \,\, points \,\, within \,\, bounds)$')
            plt.xlim(lower_bound-0.25*admissible_range, upper_bound+0.25*admissible_range)
            plt.ylim(0.0, 1.25*N)
            plt.legend(loc='best')
            # save plot and close
            print('\n\t'+'saving final image...', end='')
            file_name = plot_name+'.png'
            plt.savefig(file_name, dpi=300)
            print('figure saved: '+plot_name)
            plt.close(plot_name)
            # open the saved image, if desired
            if auto_open:
                webbrowser.open(file_name)
    return all_omegas_valid
#-----------------------------------------------------------------------------#
def solve_HB_problem(omegas, time_discretization, the_ode, delta_tau, 
                     constant_init_guess, residual_convergence_criteria, 
                     make_plot=False, auto_open_plot=False, make_movie=False, 
                     auto_play_movie=False, verbose=True, 
                     optimize_omegas=False):
    '''
    this subroutine returns a convergered harmonic-balance solution comprising
    the K angular frequencies given.
    Inputs:
        - omegas: list of K angular frequencies to build up the HB solution
        - time_discretization: 'use_T1' (equally spaced points just spanning 
                                the period corresponding to the lowest omega)
                                                  OR
                               'use_Nyquist' (use the time interval needed to 
                                just capture bandwidth, i.e. highest omega)
        - the_ode: instance of governing_ode class that define the ODE
        - constant_init_guess: constant-value guess for the HB initial solution
        - delta_tau: pseudo-time step size
        - residual_convergence_criteria: residual convergence criteria
        - make_plot: plot the converged solution and the residual history
        - auto_open: automatically open the plot
        - make_movie: animate the convergence process
        - auto_play: automatically open and start playing the movie
        - verbose: print residual convergence history to the screen
        - optimize_omegas: use time-accurate comparisons, trigonometric 
                           interpolation to define a cost and then use gradient
                           descent in conjunction with analytically computed
                           derivatives of the interpolant to change the values
                           of the angular frequencies in such a way that 
                           minimizes the cost.
    Output:
        - t_HB: the time instances over which the HB solution is defined
        - f_HB: the converged harmonic-balance solution
    '''
    import numpy as np
    import math
    from matplotlib import pyplot as plt
    # if using the T1 time discretization, check to see if all the given 
    # angular frequencies are valid, if not, print a warning to the screen
    if time_discretization == 'use_T1':
        all_omegas_valid = HB_omega_check(omegas, make_plots=False)
        if not all_omegas_valid:
            print('\n\tWARNING: THERE ARE INADMISSIBLE OMEGAS!')
    # maximum number of pseudo-time steps to try (can be changed, if needed)
    max_pseudo_steps = 10000000
    # print message to the screen
    if verbose:
        print('computing the harmonic-balance solution...')
    else:
        print('\tcomputing the harmonic-balance solution...', end ='')
    # create the harmonic balance operator matrix and find the time instances
    D_HB, t_HB = harmonic_balance_operator(omegas, time_discretization)
    # create a constant-valued initial guess for the HB solution
    f_HB = np.array([constant_init_guess]*len(t_HB)).reshape((len(t_HB),1))
    #f_HB = init_guess+np.random.rand(len(t_HB)).reshape((len(t_HB),1))
    # create a list for the solution evolution history
    f_HB_history = [np.copy(f_HB)]
    # create a list for the residual evolution history
    residual_history = []
    
    # set some preliminaries if changing omegas w/ gradient descent
    if optimize_omegas:
        
        # learning rate for gradient descent
        eta = 1e-6
        # level of partial convergence at which to begin optimizing the omegas 
        # (value, 'almost full convergence')
        partial_convergence_level = 1e-1
        partial_convergence_level = 'almost full convergence'
        # set where to start ('first instance', 'last instance', 'both ends')
        start_time_marching_at = 'first instance'
        #start_time_marching_at = 'last instance'
        start_time_marching_at = 'both ends'        
        # specify which cost function to use (1 = curve, 2=derivative, 3=both)
        use_cost_number = 3
        # the length of the time-accurate segment, as a percent of T1
        percent_T1_spanned = 50.0
        # exponentially scale the cost function 
        exponentially_scale_cost = True
        # cauchy criterion for the cost
        cauchy_criterion = 1e-6
        # "velocity" coefficient for momentum gradient descent, in range (0,1]
        # gamma = 0 : regular gradient descent with learning rate eta
        # gamma = 1 : current gradient is added to cummulative sum of all the 
        #             gradients found from previous iterations
        # 0 < gamma < 1 : current gradient is added to all the previous
        #                 gradients, but with decreasing weights the farther 
        #                 they get from the current iteration
        gamma = 0.7
        
        # compute the time interval between HB time in
        t_HB_interval = t_HB[1]
        # set the time step for the time-accurate steps
        delta_t = t_HB_interval/50.0
        # no. of comparison points between interpolant and time-accurate points
        n_comp_points = int((percent_T1_spanned/100.0)*((2.0*np.pi/omegas[0])/delta_t - 1))
        # set the value to use "full convergence," using machine zero
        if partial_convergence_level == 'almost full convergence':
            machine_zero = sys.float_info.epsilon
            fully_converged = 1e3*machine_zero
            almost_fully_converged = 10*fully_converged
            partial_convergence_level = almost_fully_converged
        # count up the number of omegas being used
        K = len(omegas)
        # define the number of time instances
        N = 2*K+1
        # intitalize the history lists
        t_HB_history = []               # time instances
        t_HB_int_history = []           # fine time grids for interpolation
        f_HB_int_history = []           # interpolations of the HB solutions
        if start_time_marching_at == 'both ends':
            t_stepped_1_history = []
            f_stepped_1_history = []
            t_stepped_2_history = []
            f_stepped_2_history = []
            dfdt_stepped_1_history = []
            f_HB_int_stepped_1_history = []
            dfdt_HB_int_stepped_1_history = []
            dfdt_stepped_2_history = []
            f_HB_int_stepped_2_history = []
            dfdt_HB_int_stepped_2_history = []
        else:
            t_stepped_history = []
            f_stepped_history = []
            dfdt_stepped_history = []
            f_HB_int_stepped_history = []
            dfdt_HB_int_stepped_history = []
        cost_history = []
        omegas_history = []
        f_HB_history_opt = []
        # initialize interpolant vectors (just so the warning goes away)
        t_HB_int = []
        f_HB_int = []
        # initialize "velocity" values for momentum gradient descent for each
        # of the K omegas
        v = [0.0]*K
        # initialize previous_cost to a very big number
        previous_cost = 1e6
        
        
    # set the flag for the optimization plotting
    make_opt_plot = False
    # turn off the adjust_omegas option (until partial convergence reached)
    currently_optimizing = False
    # start the pseudo-transient continuation method
    for iteration in range(max_pseudo_steps):
        # if we've started optimizing then recompute the HB basis
        if currently_optimizing:            
            # recompute the operator matrix and the time instances
            D_HB, t_HB = harmonic_balance_operator(omegas, time_discretization)
            # interpolate the previous solution onto these new time instances
            t_HB, f_HB = linearInterp(t_HB_int, f_HB_int, t_HB)
            # recast the solution as a numpy array
            f_HB = np.array(f_HB)
            # make it a column vector
            f_HB = np.reshape(f_HB,(N,1))
        # compute the residual vector corresponding the current solution
        func_evaluations = the_ode.evaluate(t_HB,f_HB)
        matrix_vector_product = np.dot(D_HB,f_HB)
        residual = func_evaluations - matrix_vector_product
        # compute the norm of the residual vector and print to the screen
        norm_residual = np.linalg.norm(residual)
        residual_history.append(norm_residual)
        if verbose and not optimize_omegas:
            print('\n\titer: '+str(iteration)+'\t||residual||: '+str(norm_residual), end='')
        # compute the "error," which is a function of the residual, for pseudo-
        # trasient continuation (pseudo-time stepping)
        I = np.eye(len(residual))     # identity matrix
        step_size = delta_tau
        B = step_size*I
        error = np.dot(B,residual)
        # if convergence criteria is met, end, else, update solution
        if norm_residual < residual_convergence_criteria:
            # converged solution found
            if verbose:
                print('\n\n\t\tharmonic balance solution found.\n')
            else:
                print('done.\n')
            break
        elif np.isnan(norm_residual) or np.isinf(norm_residual):
            # unstable solution
            if verbose:
                print('\n\n\t\tunstable solution. try again.\n')
            else:
                print('unstable solution. try again.\n')
            break
        else:
            # time-march solution at each time instance by one delta_t. let
            # discrepancy between time-marched point and tonal interpolation
            # represent cost. take derivatives of cost w.r.t. each omega.
            # use derivate multiplied by a learning rate to minimize cost using 
            # gradient descent
            if optimize_omegas and norm_residual < partial_convergence_level:
                # turn on the switch signifying that optimizing is happening
                currently_optimizing = True
                # interpolate the HB solution using the current omegas
                t_HB_int, f_HB_int, \
                dfdt_HB_int, a, b = trig_interp_given_freqs(t_HB, f_HB, omegas,
                                                              x_int=np.linspace(t_HB[0],2.0*t_HB[-1],t_HB[-1]/delta_t),
                                                              return_coeffs=True)
                # assume the solution found at each time instance is the 
                # initial condition for a new time-accurate problem. take one 
                # time-marched step and record the value
                if start_time_marching_at == 'both ends':
                    t_stepped_1 = []
                    f_stepped_1 = []
                    t_stepped_2 = []
                    f_stepped_2 = []
                else:
                    t_stepped = []
                    f_stepped = []                
                # run through the time comparison points
                for i in range(n_comp_points):
                    # take a time-accurate step from the last one, starting at 0
                    if i == 0:
                        # set the "initial condition" for the time-marched bit
                        if start_time_marching_at == 'first instance':
                            # solution at the first time instance
                            f_stepped_i = copy.copy(f_HB[0])
                            # first time instance (always 0.0)
                            t_stepped_i = copy.copy(t_HB[0])
                        if start_time_marching_at == 'last instance':
                            # solution at the last time instance
                            f_stepped_i = copy.copy(f_HB[-1])
                            # last time instance
                            t_stepped_i = copy.copy(t_HB[-1])
                        if start_time_marching_at == 'both ends':
                            f_stepped_1_i = copy.copy(f_HB[0])
                            t_stepped_1_i = copy.copy(t_HB[0])
                            f_stepped_2_i = copy.copy(f_HB[-1])
                            t_stepped_2_i = copy.copy(t_HB[-1])
                    else:
                        if start_time_marching_at == 'both ends':
                            f_stepped_1_i += delta_t*the_ode.evaluate(t_stepped_1_i,f_stepped_1_i[0])
                            t_stepped_1_i += delta_t
                            f_stepped_2_i += delta_t*the_ode.evaluate(t_stepped_2_i,f_stepped_2_i[0])
                            t_stepped_2_i += delta_t
                        else:
                            # advance the solution by taking a time step
                            f_stepped_i += delta_t*the_ode.evaluate(t_stepped_i,f_stepped_i[0])
                            t_stepped_i += delta_t
                    if start_time_marching_at == 'both ends':
                        t_stepped_1.append(t_stepped_1_i)
                        f_stepped_1.append(f_stepped_1_i[0])
                        t_stepped_2.append(t_stepped_2_i)
                        f_stepped_2.append(f_stepped_2_i[0])
                    else:
                        # record the stepped time points
                        t_stepped.append(t_stepped_i)
                        # record the time-marched value
                        f_stepped.append(f_stepped_i[0])
                # compute the first derivative of the time-marched curve (using 
                # central differences, i.e. exluding the first and last points)
                if start_time_marching_at == 'both ends':
                    dfdt_stepped_1 = []
                    dfdt_stepped_2 = []
                else:
                    dfdt_stepped = []
                for i in range(n_comp_points):
                    if i == 0 or i == n_comp_points-1:
                        # at the first and last points, append a nan
                        if start_time_marching_at == 'both ends':
                            dfdt_stepped_1.append('nan')
                            dfdt_stepped_2.append('nan')
                        else:
                            dfdt_stepped.append('nan')
                    else:
                        # take a central difference and store the result
                        if start_time_marching_at == 'both ends':
                            dfdt_stepped_1_i = (f_stepped_1[i+1]-f_stepped_1[i-1])/(2.0*delta_t)
                            dfdt_stepped_1.append(dfdt_stepped_1_i)
                            dfdt_stepped_2_i = (f_stepped_2[i+1]-f_stepped_2[i-1])/(2.0*delta_t)
                            dfdt_stepped_2.append(dfdt_stepped_2_i)
                        else:
                            dfdt_stepped_i = (f_stepped[i+1]-f_stepped[i-1])/(2.0*delta_t)
                            dfdt_stepped.append(dfdt_stepped_i)
                # use linear interpolation to recover the values of the 
                # interpolant and the interpolant's derivative at the stepped
                # locations
                if start_time_marching_at == 'both ends':
                    f_HB_int_stepped_1 = []
                    dfdt_HB_int_stepped_1 = []
                    f_HB_int_stepped_2 = []
                    dfdt_HB_int_stepped_2 = []
                else:
                    f_HB_int_stepped = []
                    dfdt_HB_int_stepped = []
                for i in range(n_comp_points):
                    if start_time_marching_at == 'both ends':
                        # point on the interpolant - first track
                        t_stepped_1_i_point, f_HB_int_stepped_1_i = linearInterp(t_HB_int, f_HB_int, [t_stepped_1[i]])
                        f_HB_int_stepped_1_i = copy.copy(f_HB_int_stepped_1_i[0])
                        f_HB_int_stepped_1.append(f_HB_int_stepped_1_i)
                        # point on the interpolant derivative - first track
                        t_stepped_1_i_point, dfdt_HB_int_stepped_1_i = linearInterp(t_HB_int, dfdt_HB_int, [t_stepped_1[i]])
                        dfdt_HB_int_stepped_1_i = copy.copy(dfdt_HB_int_stepped_1_i[0])
                        dfdt_HB_int_stepped_1.append(dfdt_HB_int_stepped_1_i)
                        # point on the interpolant - second track
                        t_stepped_2_i_point, f_HB_int_stepped_2_i = linearInterp(t_HB_int, f_HB_int, [t_stepped_2[i]])
                        f_HB_int_stepped_2_i = copy.copy(f_HB_int_stepped_2_i[0])
                        f_HB_int_stepped_2.append(f_HB_int_stepped_2_i)
                        # point on the interpolant derivative - second track
                        t_stepped_2_i_point, dfdt_HB_int_stepped_2_i = linearInterp(t_HB_int, dfdt_HB_int, [t_stepped_2[i]])
                        dfdt_HB_int_stepped_2_i = copy.copy(dfdt_HB_int_stepped_2_i[0])
                        dfdt_HB_int_stepped_2.append(dfdt_HB_int_stepped_2_i)
                    else:
                        # use linear interpolation to recover the value of the 
                        # tonal interpolant at the i-th stepped time point
                        t_stepped_i_point, f_HB_int_stepped_i = linearInterp(t_HB_int, f_HB_int, [t_stepped[i]])
                        # pull the value out of the array
                        f_HB_int_stepped_i = copy.copy(f_HB_int_stepped_i[0])
                        # append to the list
                        f_HB_int_stepped.append(f_HB_int_stepped_i)  
                        # use linear interpolation to recover the value of the time
                        # derivative of the tonal interpolant at the i-th 
                        # stepped time point
                        t_stepped_i_point, dfdt_HB_int_stepped_i = linearInterp(t_HB_int, dfdt_HB_int, [t_stepped[i]])
                        # pull the value out of the array
                        dfdt_HB_int_stepped_i = copy.copy(dfdt_HB_int_stepped_i[0])
                        # append to the list
                        dfdt_HB_int_stepped.append(dfdt_HB_int_stepped_i)  
                # cost #1: sum of mean-squared errors of the function values
                costs_1 = []
                for i in range(n_comp_points):
                    if start_time_marching_at == 'both ends':
                        cost_1_i = (f_stepped_1[i]-f_HB_int_stepped_1[i])**2.0 + (f_stepped_2[i]-f_HB_int_stepped_2[i])**2.0
                    else:
                        cost_1_i = (f_stepped[i]-f_HB_int_stepped[i])**2.0
                    costs_1.append(cost_1_i)
                cost_1 = sum(costs_1)
                # cost #2: sum the mean-squared errors of the derivative values
                costs_2 = []
                for i in range(n_comp_points):
                    if i==0 or i==n_comp_points-1:
                        cost_2_i = 0.0
                    else:
                        if start_time_marching_at == 'both ends':
                            cost_2_i = (dfdt_stepped_1[i]-dfdt_HB_int_stepped_1[i])**2.0 + (dfdt_stepped_2[i]-dfdt_HB_int_stepped_2[i])**2.0
                        else:
                            cost_2_i = (dfdt_stepped[i]-dfdt_HB_int_stepped[i])**2.0
                    costs_2.append(cost_2_i)
                cost_2 = sum(costs_2)
                # cost #3: sum costs 1 and 2 (excluding the end points)
                costs_3 = []
                for i in range(1,n_comp_points-1):
                    cost_3_i = costs_1[i] + costs_2[i]
                    costs_3.append(cost_3_i)
                cost_3 = sum(costs_3)
                # record the cost from this time instance
                if use_cost_number == 1:
                    cost = cost_1
                elif use_cost_number == 2:
                    cost = cost_2
                else:
                    cost = cost_3
                # if desired, exponentially scale the cost
                if exponentially_scale_cost:
                    cost = np.exp(cost)
                # print the cost to the console and record it
                print('\n\tcost =', cost)
                cost_history.append(cost)
                # check cauchy condition to see if optimization should go on
                if abs(cost - previous_cost) <= cauchy_criterion:
                    optimize_omegas = False
                    make_opt_plot = True
                else:  
                    previous_cost = cost
                # record the location of the time instances for this iteration
                t_HB_history.append(t_HB)
                # record the solution values at the current instances
                f_HB_history_opt.append(f_HB)
                # record the times and function values of the interpolation
                t_HB_int_history.append(t_HB_int)
                f_HB_int_history.append(f_HB_int)
                if start_time_marching_at == 'both ends':
                    t_stepped_1_history.append(t_stepped_1)
                    f_stepped_1_history.append(f_stepped_1)
                    t_stepped_2_history.append(t_stepped_2)
                    f_stepped_2_history.append(f_stepped_2)
                    dfdt_stepped_1_history.append(dfdt_stepped_1)
                    dfdt_stepped_2_history.append(dfdt_stepped_2)
                    f_HB_int_stepped_1_history.append(f_HB_int_stepped_1)
                    f_HB_int_stepped_2_history.append(f_HB_int_stepped_2)
                    dfdt_HB_int_stepped_1_history.append(dfdt_HB_int_stepped_1)
                    dfdt_HB_int_stepped_2_history.append(dfdt_HB_int_stepped_2)
                else:
                    # record the location of the stepped points for this instance 
                    # (these shouldn't change if only stepping forward from 0.0)
                    t_stepped_history.append(t_stepped)
                    # record the time-marched values
                    f_stepped_history.append(f_stepped)
                    # record the derivatives of the time-marched values
                    dfdt_stepped_history.append(dfdt_stepped)
                    # record the interpolant values at the stepped points
                    f_HB_int_stepped_history.append(f_HB_int_stepped)
                    # record the interpolant-derivative values at stepped points
                    dfdt_HB_int_stepped_history.append(dfdt_HB_int_stepped)
                # record the omegas used to compute the previous solution
                omegas_history.append(omegas)
                print('\n\tomegas =', str(omegas)[1:-1])
                # update each of the K omegas using gradient descent
                for k in range(0,K):
                    # compute the derivative of the first cost function with
                    # respect to the k-th angular frequency
                    cost_1_derivatives = []
                    for i in range(n_comp_points):
                        if start_time_marching_at == 'first instance':
                            cost_1_derivative_i = 2.0*i*delta_t* \
                                        (f_stepped[i]-f_HB_int_stepped[i])* \
                                        (a[k+1]*math.sin(omegas[k]*i*delta_t) \
                                        - b[k+1]*math.cos(omegas[k]*i*delta_t))
                        if start_time_marching_at == 'last instance':
                            # for omega_1
                            if k==0:                                
                                cost_1_derivative_i = 2.0*i*delta_t* \
                                        (f_stepped[i]-f_HB_int_stepped[i])* \
                                        (a[k+1]*math.sin(omegas[k]*t_stepped[i]) \
                                        - b[k+1]*math.cos(omegas[k]*t_stepped[i]))
                            # for all the other omegas
                            else:                                
                                cost_1_derivative_i = 2.0*t_stepped[i]* \
                                        (f_stepped[i]-f_HB_int_stepped[i])* \
                                        (a[k+1]*math.sin(omegas[k]*t_stepped[i]) \
                                        - b[k+1]*math.cos(omegas[k]*t_stepped[i]))
                        if start_time_marching_at == 'both ends':
                            # for omega_1
                            if k==0:
                                cost_1_derivative_i = 2.0*i*delta_t* \
                                (f_stepped_1[i]-f_HB_int_stepped_1[i])* \
                                        (a[k+1]*math.sin(omegas[k]*i*delta_t) \
                                        - b[k+1]*math.cos(omegas[k]*i*delta_t)) \
                                                    + 2.0*i*delta_t* \
                                                    (f_stepped_2[i]-f_HB_int_stepped_2[i])* \
                                        (a[k+1]*math.sin(omegas[k]*t_stepped_2[i]) \
                                        - b[k+1]*math.cos(omegas[k]*t_stepped_2[i]))
                            # for all the other omegas
                            else:
                                cost_1_derivative_i = 2.0*i*delta_t* \
                                        (f_stepped_1[i]-f_HB_int_stepped_1[i])* \
                                        (a[k+1]*math.sin(omegas[k]*i*delta_t) \
                                        - b[k+1]*math.cos(omegas[k]*i*delta_t)) \
                                                    + 2.0*t_stepped_2[i]* \
                                        (f_stepped_2[i]-f_HB_int_stepped_2[i])* \
                                        (a[k+1]*math.sin(omegas[k]*t_stepped_2[i]) \
                                        - b[k+1]*math.cos(omegas[k]*t_stepped_2[i]))
                        # record the derivative of this omega    
                        cost_1_derivatives.append(cost_1_derivative_i)
                    dcost1_domega_k = sum(cost_1_derivatives)
                    # compute the derivative of the second cost function with
                    # respect to the k-th angular frequency
                    cost_2_derivatives = []
                    for i in range(n_comp_points):
                        if i==0 or i==n_comp_points-1:
                            cost_2_derivative_i = 0.0
                        else:
                            if start_time_marching_at == 'first instance':
                                cost_2_derivative_i = 2.0*(dfdt_stepped[i] - dfdt_HB_int_stepped[i]) \
                                                    *((b[k+1]-omegas[k]*i*delta_t*a[k+1])*math.cos(omegas[k]*i*delta_t) \
                                                      -(a[k+1]+omegas[k]*i*delta_t*b[k+1])*math.sin(omegas[k]*i*delta_t))
                            if start_time_marching_at == 'last instance':
                                # for the first omega
                                if k==0:
                                    cost_2_derivative_i = 2.0*(dfdt_stepped[i] - dfdt_HB_int_stepped[i]) \
                                                    *((a[k+1]+omegas[k]*b[k+1]*i*delta_t)*math.sin(omegas[k]*t_stepped[i]) \
                                                     +(omegas[k]*a[k+1]*i*delta_t-b[k+1])*math.cos(omegas[k]*t_stepped[i]))
                                # for all the others
                                else:
                                    cost_2_derivative_i = 2.0*(dfdt_stepped[i] - dfdt_HB_int_stepped[i]) \
                                                    *((a[k+1]+omegas[k]*b[k+1]*t_stepped[i])*math.sin(omegas[k]*t_stepped[i]) \
                                                     +(omegas[k]*a[k+1]*t_stepped[i]-b[k+1])*math.cos(omegas[k]*t_stepped[i]))
                            if start_time_marching_at == 'both ends':
                                # for the first omega
                                if k==0:
                                    cost_2_derivative_i = 2.0*(dfdt_stepped_1[i] - dfdt_HB_int_stepped_1[i]) \
                                                    *((b[k+1]-omegas[k]*i*delta_t*a[k+1])*math.cos(omegas[k]*i*delta_t) \
                                                      -(a[k+1]+omegas[k]*i*delta_t*b[k+1])*math.sin(omegas[k]*i*delta_t)) \
                                                        + 2.0*(dfdt_stepped_2[i] - dfdt_HB_int_stepped_2[i]) \
                                                    *((a[k+1]+omegas[k]*b[k+1]*i*delta_t)*math.sin(omegas[k]*t_stepped_2[i]) \
                                                     +(omegas[k]*a[k+1]*i*delta_t-b[k+1])*math.cos(omegas[k]*t_stepped_2[i]))
                                # for all the others
                                else:
                                    cost_2_derivative_i = 2.0*(dfdt_stepped_1[i] - dfdt_HB_int_stepped_1[i]) \
                                                    *((b[k+1]-omegas[k]*i*delta_t*a[k+1])*math.cos(omegas[k]*i*delta_t) \
                                                      -(a[k+1]+omegas[k]*i*delta_t*b[k+1])*math.sin(omegas[k]*i*delta_t)) \
                                                        + 2.0*(dfdt_stepped_2[i] - dfdt_HB_int_stepped_2[i]) \
                                                    *((a[k+1]+omegas[k]*b[k+1]*t_stepped_2[i])*math.sin(omegas[k]*t_stepped_2[i]) \
                                                     +(omegas[k]*a[k+1]*t_stepped_2[i]-b[k+1])*math.cos(omegas[k]*t_stepped_2[i]))
                        cost_2_derivatives.append(cost_2_derivative_i)
                    dcost2_domega_k = sum(cost_2_derivatives)
                    # compute the derivative of the third cost function with 
                    # respect to the k-th angular frequency (add them)
                    dcost3_domega_k = dcost1_domega_k + dcost2_domega_k
                    # use the desired gradient component
                    if use_cost_number == 1:
                        dC_domega_k = dcost1_domega_k
                    elif use_cost_number == 2:
                        dC_domega_k = dcost2_domega_k
                    else:
                        dC_domega_k = dcost3_domega_k
                    # if desired, use the exponentially scaled cost gradient
                    if exponentially_scale_cost:
                        dC_domega_k = dC_domega_k*cost
                    # set "velocity" value for momentum gradient descent
                    v[k] = gamma*v[k] + eta*dC_domega_k
                    # update the value of omega using momentum gradient descent
                    omegas[k] = omegas[k] - v[k]
                # sort them in ascending order
                omegas = sorted(omegas)
                # if using the T1 time discretization, check to see if these 
                # angular frequencies are valid, if not, print a warning
                if time_discretization == 'use_T1':
                    all_omegas_valid = HB_omega_check(omegas, make_plots=False)
                    # if there are invalid frequencies, end the simulation
                    if not all_omegas_valid:
                        break
            else:
                # turn off the switch signifying that optimizing is happening
                # (don't start optimizing until partial convergence is reached)
                currently_optimizing = False
            # update the solution
            f_HB += error
            # append the updated solution to the solution history
            f_HB_history.append(np.copy(f_HB))
        # if we've reached the maximum allowable number of pseudo-time steps
        if iteration == max_pseudo_steps-1:
            print('\n\t\tmaximum number of pseudo-time steps reached.\n')
    # if animation desired but no plot, make plot anyway and print warning
    if make_movie and not make_plot:
        print('\n\tcan\'t make movie without making plot...making plot.')
        make_plot = True
    # if desired, plot the results
    if make_plot:
        # plotting: USER INPUTS! want to animate the solution history or just
        # plot the final result? (True=animate, False=just print final result)
        animate_plot = make_movie
        plot_name = 'harmonic-balance ODE'
        n_images = iteration+1          # total number of images computed
        skip_images = 100               # images to skip between frames
        auto_play = auto_play_movie     # automatically play the movie
        auto_open = auto_open_plot      # automatically open the final image
        # plotting: instantiate the figure
        fig = plt.figure(plot_name)
        # plotting: rescale the figure window to fit both subplots
        xdim, ydim = plt.gcf().get_size_inches()
        # for two plots, this scaling can't be more than 1.7!!!
        plt.gcf().set_size_inches(1.7*xdim, ydim, forward=True)
        # set the title for the HB solution plot
        title = ''
        counter=1
        for omega in omegas:
            title = title + '$\omega_{'+str(counter)+'} ='+str(omega)+'\quad $'
            counter += 1
        # things that won't change for the residual history plot
        plt.subplot(1,2,2)
        plt.xlabel('$iteration$', fontsize=16)
        plt.ylabel('$\\left\Vert \\frac{\partial f}{\partial t} \minus D_{HB}f_{HB} \\right\Vert_2$', fontsize=16)
        plt.title(r'$\Delta\tau = '+str(delta_tau)+'$')
        plt.xlim(0,iteration)
        min_power = int(math.log(min(residual_history),10))-1
        max_power = int(math.log(max(residual_history),10))+1
        plt.ylim(pow(10,min_power), pow(10,max_power))
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
                plt.subplot(1,2,1)
                plt.cla()
                plt.plot(t_HB,f_HB_history[n],'mo')
                t_HB_int, f_HB_int, dummy = trig_interp_given_freqs(t_HB, f_HB_history[n], omegas)
                plt.plot(t_HB_int, f_HB_int, 'm--')
                plt.xlabel('$t$', fontsize=16)
                plt.ylabel('$f_{HB}$', fontsize=16)
                plt.ylim(np.min(f_HB_history), np.max(f_HB_history))
                #plt.title(title)
                # plot the residual
                plt.subplot(1,2,2)
                if n > 0 and residual_history[n] >= residual_history[0]:
                    plt.semilogy(residual_history[:n+1],'g-')
                else:
                    plt.semilogy(residual_history[:n+1],'r-')
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
        plt.subplot(1,2,1)
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
        # if desired, plot the process of optimization of angular frequencies
        if make_opt_plot:
            # plotting: USER INPUTS! want to animate the time-accurate or just
            # plot the final result? (True=animate, False=just print final result)
            animate_plot = make_movie
            plot_name = 'omega optimization'
            n_images = len(omegas_history)  # total number of images computed
            skip_images = 25                 # images to skip between frames
            auto_play = auto_play_movie     # automatically play the movie
            auto_open = auto_open_plot      # automatically open the final image
            # plotting: instantiate the figure
            fig = plt.figure(plot_name,figsize=(10,10))
            # white space for the first plot
            f_white_space = max(f_HB_int_history[0])-min(f_HB_int_history[0])
            # things that won't change for the cost history plot
            plt.subplot(3,1,2)
            plt.xlabel('$optimizing \,\, iteration$', fontsize=16)
            plt.ylabel('$C$', fontsize=16)
            plt.xlim(0.0,len(omegas_history)-1.0)
            white_space = 0.25*(max(cost_history)-min(cost_history))
            plt.ylim(min(cost_history)-white_space, max(cost_history)+white_space)
            # create the appropriate title for the cost plot
            cost_title = '$C = '
            if exponentially_scale_cost:
                cost_title += 'exp('
            if start_time_marching_at == 'first instance':
                cost_1_title = '\sum_{i=0}^{n_{cp}-1} [ f_i - \\tilde{f}(i\Delta t)]^2'
                cost_2_title = '\sum_{l=1}^{n_{cp}-2} [ \dot{f}_l - \dot{\\widetilde{f}}(l\Delta t)]^2'
            if start_time_marching_at == 'last instance':
                cost_1_title = '\sum_{i=0}^{n_{cp}-1} [ f_i - \\tilde{f}(t_{N-1}^{HB}+i\Delta t)]^2'
                cost_2_title = '\sum_{l=1}^{n_{cp}-2} [ \dot{f}_l - \dot{\\widetilde{f}}(t_{N-1}^{HB}+l\Delta t)]^2'
            if start_time_marching_at == 'both ends':
                cost_1_title = '\sum_{i=0}^{n_{cp}-1} [ f_i^{(1)} - \\tilde{f}(i\Delta t)]^2 + [ f_i^{(2)} - \\tilde{f}(t_{N-1}^{HB}+i\Delta t)]^2'
                cost_2_title = '\sum_{l=1}^{n_{cp}-2} [ \dot{f}_l^{(1)} - \dot{\\widetilde{f}}(l\Delta t)]^2 + [ \dot{f}_l^{(2)} - \dot{\\widetilde{f}}(t_{N-1}^{HB}+l\Delta t)]^2'
            cost_3_title = cost_1_title + '+' + cost_2_title
            if use_cost_number==1:
                cost_title += cost_1_title
            elif use_cost_number==2:
                cost_title += cost_2_title
            else:
                cost_title += cost_3_title
            if exponentially_scale_cost:
                cost_title += ')'
            cost_title += '$'
            plt.title(cost_title, y=1.07)
            # things that won't be changing in the omegas plot
            plt.subplot(3,1,3)
            omega_colors = ['orange', 'g', 'b', 'm', 'brown', 'r', 'k']
            # create histories for each individual omega, for the omegas plot
            individual_omega_histories = [[] for k in range(K)]
            for omegas_set in omegas_history:
                counter = 0
                for omega_k in omegas_set:
                    individual_omega_histories[counter].append(omega_k)
                    counter += 1
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
                    plt.subplot(3,1,1)
                    plt.cla()
                    plt.plot(t_HB_int_history[n],f_HB_int_history[n],'y-', label='$\\widetilde{f}$')
                    plt.plot(t_HB_history[n],f_HB_history_opt[n],'ko')
                    for index in range(n_comp_points):
                        if start_time_marching_at == 'both ends':
                            plt.plot([t_stepped_1_history[n][index]]*2, [f_stepped_1_history[n][index], f_HB_int_stepped_1_history[n][index]],'c-')
                            plt.plot([t_stepped_2_history[n][index]]*2, [f_stepped_2_history[n][index], f_HB_int_stepped_2_history[n][index]],'c-')
                        else:
                            plt.plot([t_stepped_history[n][index]]*2, [f_stepped_history[n][index], f_HB_int_stepped_history[n][index]],'c-')
                    if start_time_marching_at == 'both ends':
                        plt.plot(t_stepped_1_history[n],f_HB_int_stepped_1_history[n],'y.')
                        plt.plot(t_stepped_1_history[n],f_stepped_1_history[n],'b.-', label='$f^{(1)}$')
                        plt.plot(t_stepped_2_history[n],f_HB_int_stepped_2_history[n],'y.')
                        plt.plot(t_stepped_2_history[n],f_stepped_2_history[n],'b.-', label='$f^{(2)}$')
                    else:
                        plt.plot(t_stepped_history[n],f_stepped_history[n],'b.-', label='$f$')
                        plt.plot(t_stepped_history[n],f_HB_int_stepped_history[n],'y.')
                    plt.xlabel('$t, [s]$', fontsize=16)
                    plt.ylabel('$f(t)$', fontsize=16)
                    if start_time_marching_at == 'both ends':
                        plt.xlim(0.0, 1.5*t_stepped_2_history[-1][-1])
                        plt.ylim(f_stepped_1_history[0][0]-f_white_space, f_stepped_1_history[0][0]+f_white_space)
                    else:
                        plt.xlim(0.0, 1.5*t_stepped_history[-1][-1])
                        plt.ylim(f_stepped_history[0][0]-f_white_space, f_stepped_history[0][0]+f_white_space)
                    plt.legend(loc='best')
                    # cost plot
                    plt.subplot(3,1,2)
                    if n > 0 and cost_history[n] >= cost_history[n-1]:
                        plt.plot(cost_history[:n+1],'g-')
                    else:
                        plt.plot(cost_history[:n+1],'r-')
                    # omega trajectories plot
                    plt.subplot(3,1,3)
                    plt.cla()
                    for counter in range(K):
                        plt.plot(range(n+1),individual_omega_histories[counter][:n+1], color=omega_colors[counter], label='$\omega_{'+str(counter)+'}$')
                        plt.plot(n, individual_omega_histories[counter][n], color=omega_colors[counter], marker='.')                      
                    plt.xlabel('$optimizing \,\, iteration$', fontsize=16)
                    plt.ylabel('$\omega_k$')
                    plt.xlim(0.0,len(omegas_history)-1.0)
                    white_space = max(omegas_history[0])-min(omegas_history[0])
                    plt.ylim(min(omegas_history[0])-0.5*white_space,max(omegas_history[0])+0.5*white_space)                    
                    #plt.legend(loc='best')
                    title = ''
                    counter = 1
                    for omega in omegas_history[n]:
                        title = title + '$\omega_{'+str(counter)+'} ='+str(np.round(omega,4))+'\quad $'
                        counter += 1
                    title += '$\quad\quad\eta = '+str(eta)+'$'
                    plt.title(title)
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
    # return the converged solution and, if computed, the frequencies found
    if make_opt_plot:
        return t_HB, f_HB, omegas
    else:
        return t_HB, f_HB
#-----------------------------------------------------------------------------#
def HB_sol_plus_DFT(B, kappa, time_discretization, the_ode, HB_initial_guess,
                    partial_convergence, AC_energy_to_capture):
    '''
    this subroutine takes in a given bandwidth and a given number of points to 
    discretize that bandwidth. the resulting discrete frequencies are used to
    construct a harmonic-balance solution and solve it to a specified level of 
    convergence. then, take the DFT of this solution, refine the peaks, and
    return the boundaries of the clustered peaks
    '''
    from time_spectral import myLinspace
    import numpy as np
    # print inputs to the screen
    print('\n\tbandwidth: \t'+str(B)+'\n\tpoints: \t'+str(kappa)+'\n')
    # create a uniformly spaced list of trial omegas (not including zero)
    trial_omegas = myLinspace(0.0, B, kappa+1)[1:]
    # solve a harmonic-balance problem using these omegas
    t_HB, f_HB = solve_HB_problem(trial_omegas, time_discretization, the_ode, 
                                    delta_tau=0.01, 
                                    constant_init_guess=HB_initial_guess, 
                                    residual_convergence_criteria=partial_convergence, 
                                    make_plot=False, auto_open_plot=False, 
                                    make_movie=False, auto_play_movie=False, 
                                    verbose=False)
    # take the DFT of this solution and find the refined peaks
    freq, F, powers, peaks_found, peak_boundaries = my_dft(t_HB, np.squeeze(f_HB), 
                                         percent_energy_AC_peaks=AC_energy_to_capture,
                                         shift_frequencies=True,
                                         use_angular_frequencies=True,
                                         plot_spectrum=True, 
                                         plot_log_scale=True,
                                         refine_peaks=True,
                                         auto_open_plot=True,
                                         verbose=False,
                                         title_suffix='\quad(\omega_{actual}='\
                                         +str(the_ode.actual_omegas)[1:-1]+')',
                                         plot_suffix=' - B='+str(round(B,2)),
                                         use_fft=True)
    # define "error bands" for these peaks as +/-(delta_freq/2)
    delta_freq = freq[-1]-freq[-2]
    error_band = delta_freq/2.0
    # record the number of refined peaks returned
    n_peaks = len(peaks_found)
    # record each peak's range (based on the error bands) as a tuple
    peak_ranges = [(peak-error_band, peak+error_band) for peak in peaks_found]
    # print the peaks found
    print('\n\t\tpeaks found (+/- '+str(round(error_band,3))+'): ')
    for peak in peaks_found:
        print('\t\t\t\t\t'+str(round(peak,3)))
    
    # set the constant intial guess for the next run equal to average of f_HB
    ave_f_HB = sum(f_HB)/len(f_HB)
    return n_peaks, peaks_found, peak_ranges, peak_boundaries, ave_f_HB
#-----------------------------------------------------------------------------#
def rouche_capelli_check(A, b, verbose=False):
    '''
    this function uses the Rouche-Capelli theorem to tell the user how many 
    solutions exist for the system Ax=b. simply put, if the rank of the matrix
    A is the same as the rank of the augmented matrix [A|b], then a solution 
    exists (i.e. the number of solutions is either one or infinity.) if the 
    rank of A happens to be the same as the number of columns in A, then A has
    full column rank and if solutions exists (i.e. if rank(A)==rank([A|b]),
    then there is only one unique solution. note that this means you can have
    less than full rank and still have a solution to Ax=b.
    with the HB expansion matrix, this can happen when the function being 
    sampled exactly contains only those frequencies that have been used to 
    build up the G matrix. in other words, the function vector, f_HB, is 
    already in the column space of G -- despite the column space not being 
    N-dimensional (the column space of G is an r-dimensional subspace within
    C^N)! 
    recall, the whole point of solving Gc=f is to find c. c is nothing more 
    than an expansion of f in the basis provided by the columns of G. so, f_HB 
    is in the range of G, even though G does not span the space of complex 
    N-dimensional vectors (C^N). in other words, G has a left nullspace. 
    because f_HB is already in the column space of G, the augmented matrix 
    A=[G|f_HB] will have the same rank as G itself. the Rouche-Capelli theorem 
    states the implications of this observation: a system Ax=b has a solution 
    iff rank(A)=rank([A|b]). if you have full column rank, then the solution is 
    unique. otherwise, there is an infinite number of solutions
    '''
    # find n, the number of columns
    (m,n) = np.shape(A)
    # construct the augmented matrix [A|b]
    A_augmented = np.hstack((A,b))
    # run the checks of Rouche-Capelli
    if verbose:
        print('\n\n\t\t*** Rouché-Capelli ***')
    rank_A = my_rank(A)
    rank_A_augmented = my_rank(A_augmented)
    if rank_A_augmented==rank_A:
        if verbose:
            print('\tA and [A|b] have the same rank:', rank_A)
        if rank_A==n:
            if verbose:
                print('\tA has full column rank!')
                print('\ttherefore: Ax=b has a unique solution.')
            N_solutions = 1
        else:
            if verbose:
                print('\tA doesn\'t have full column rank.')
                print('\ttherefore: Ax=b has an infinite number of solutions.')
            N_solutions = np.inf
    else:
        if verbose:
            print('\tA and [A|b] have the different rank.')
            print('\t - rank(A) =', rank_A)
            print('\t - rank([A|b]) =', rank_A_augmented)
            print('\ttherefore: Ax=b has no solution!')
        N_solutions = 0
    if verbose:
        print('\t\t*** *** *** *** *** ***\n')
    # return the number of solutions to Ax=b
    return N_solutions
#-----------------------------------------------------------------------------#
def main():
    '''
    main function, executed when run as a standalone file
    '''
    # turn off interactive mode, so that plot windows don't pop up
    plt.ioff()
    
    ###############
    # user inputs #
    ###############    
    
    # TESTING Ekici & Hall (AIAA Journal, 2008)
    # they say to use 3K+1 and the pseudoinverse of G
    # [REMEMBER to turn on: time_discretization = 'use_T1-'+str(3*K+1)]!!!
    # pick two random "excitation frequencies"
    freq_1 = 1.0
    freq_2 = 8.0
    # use frequency set specified in Table 1 to build up the operator matrix,
    # namely: {f1, 2*f1, f2, 2*f2, f1+2f2, f1+f2} (harmonics and "cross terms")
    # note: Hall's "excitation frequencies" are not the same thing as my
    #       "actual frequencies." the excitation frequencies are known, 
    #       mechanical constants. by building up his operator with a set that
    #       includes harmonics and cross terms, Hall is saying that the actual
    #       frequencies experienced by the flow are more than just the two
    #       known forcing frequencies
    actual_omegas = sorted([freq_1, 2.0*freq_1, freq_2, 2.0*freq_2, freq_1 + 2.0*freq_2, freq_1 + freq_2])
    #actual_omegas = sorted([freq_1, 2.0*freq_1, freq_2, 2.0*freq_2])
    
    
    # angular frequencies in the underlying signal
    actual_omegas = [1.0, 2.5]
    #actual_omegas = [1.5, 3.75]             # 3.75 is inadmissible here!
    #actual_omegas = [1.0, 6.0]
    
    actual_omegas = [1.0, 2.1, 3.6]         # all freqs admissible
    
    #actual_omegas = [1.0, 2.1, 3.5]         # 3.5 is inadmissible here! (2K+1) (rank goes down by 1)(m=1, odd!) (case I.A)
    #actual_omegas = [1.0, 2.1, 7.0]         # 7.0 is inadmissible here! (2K+1) (rank goes down by 2)(m=2, even!) (case I.B)
    #actual_omegas = [1.0, 2.1, 4.9]         # 2.1 or 4.9 are inadmissible here! (2K+1) (rank goes down by 2) (case II)
    #actual_omegas = [1.0, 2.1, 9.1]         # 2.1 or 9.1 are inadmissible here! (2K+1) (rank goes down by 2)
    #actual_omegas = [1.0, 8.0, 9.1]
    #actual_omegas = [1.0, 2.1, 6.0]
    #actual_omegas = [1.0, 7.0, 10.5]        # this set is actually inadmissible under two cases at once! (rank goes down by 3)
    
    #actual_omegas = [1.0, 2.1, 4.0]         # 4.0 is inadmissible here! (K+1)
    
    #actual_omegas = [1.0, 2.1, 5.0]         # 5.0 is inadmissible here! (3K+1) (rank goes down by 1)(m=1, odd!)
    #actual_omegas = [1.0, 2.1, 10.0]        # 10.0 is inadmissible here! (3K+1) (rank goes down by 2)(m=2, even!)
    #actual_omegas = [1.0, 2.1, 7.9]         # 2.1 or 7.9 are inadmissible here! (3K+1) (rank goes down by 2)
    
    #actual_omegas = [1.0, 2.1, 5.0]         # 5.0 is inadmissible here! (3K+1 & K+1)
    #actual_omegas = [1.3, 2.1, 3.6, 4.7]
    #actual_omegas = [1.3, 2.1, 3.6, 5.85]   # 5.85 is inadmissble here!
    #actual_omegas = [1.0, 2.1, 17.5]
    
    #actual_omegas = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]

    # testing with periodic signals (for UTRC question)    
    #actual_omegas = [2.5, 5.0, 7.5]     # "fully periodic"
    #actual_omegas = [2.5, 5.0, 7.5, 10.0]
    #actual_omegas = [5.0, 7.5]          # "harmonically related"
    
    # angular frequencies input by the user to the HB method
    omegas = copy.copy(actual_omegas)       # use the exact values
    
    # quick-'n'-dirty noise study (use to compare errors between Nyquist and T1 under noise)
    #actual_omegas = [1.0, 2.1, 3.55]    # these are the real freqs
    #omegas = [1.0, 2.1, 3.5]            # here, we are underestimating the last freq by ~1.41% (these frequencies are also inadmissibe under T1!)
    
    
    # number of angular frequencies being used
    K = len(omegas)
    
    # select the time discretization to use for specifying the of time instances
    time_discretization = 'use_Nyquist'
    #time_discretization = 'use_Nyquist-random'
    #time_discretization = 'use_Nyquist-'+str(K+1)
    #time_discretization = 'use_Nyquist-random-'+str(K+1)
    #time_discretization = 'use_Nyquist-'+str(3*K+1)
    #time_discretization = 'use_Nyquist-random-'+str(3*K+1)
    #time_discretization = 'use_T1'
    #time_discretization = 'use_T1-random'
    #time_discretization = 'use_T1-'+str(K+1)
    #time_discretization = 'use_T1-random-'+str(K+1)
    #time_discretization = 'use_T1-'+str(3*K+1)
    #time_discretization = 'use_T1-random-'+str(3*K+1)
    
    #time_discretization = 'use_Nyquist-random-'+str(4*K+1)
    #time_discretization = 'use_T1-'+str(2*K+2)
    
    ##############################################################
    # define the governing equation based on these actual omegas #
    ##############################################################
    
    # instantiate an object of the governing equation
    the_ode = governing_ode(actual_omegas)
    
    ##############################################################################
    # inadmissibiltiy check: are these omegas compatible with T1 discretization? #
    ##############################################################################
    if 'use_T1' in time_discretization:
        # find the number of time instances being used
        try:
            # extract the number of time instances by parsing the string
            N = int(time_discretization.replace('use_T1','').replace('random','').replace('-',''))
        except ValueError:
            # no number has been found in the string, running default case 2K+1
            N = 2*K+1
        # omega_dicts contains all the constraints on each given angular 
        # frequency, invalid_omegas is a list of tuples containing the number 
        # and value of each inadmissible angular frequency
        all_omegas_valid = HB_omega_check(omegas, N)
        # if there's an invalid omega, print a warning
        if not all_omegas_valid:
            print('\n\t\t* uh oh! need to change the time discretization! *')
    else:
        # if you're not using T1 discretization
        all_omegas_valid = True
    
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
    
    # create the harmonic balance operator matrix
    print('\n( time_discretization = ', time_discretization,')')
    D_HB, t_HB = harmonic_balance_operator(omegas, time_discretization)
    
    # define an "exact" fine time grid
    T_lowest_omega = 2.0*np.pi/min(omegas)
    t_fine_points = 200
    if t_HB[-1] > T_lowest_omega:
        t = np.linspace(0.0, t_HB[-1], t_fine_points)
    else:
        t = np.linspace(0.0, T_lowest_omega, t_fine_points)
    # find the exact function and function derivative values at the time points
    f,df = my_non_periodic_fun(t, actual_omegas, verbose=True)



    # maybe add this to operator function???

    # nyquist rate for given omegas (this is a bandlimited signal)
    nyquist_rate = max(omegas)/np.pi    
    # maximum allowable time interval while still satisfying nyquist rate
    delta_t_nyquist = 1.0/nyquist_rate
    # run through the time instances and check the length of the intervals
    intervals_satisfy_nyquist = []
    nyquist_satisfied_for_all = True
    for i in range(len(t_HB)-1):
        current_delta_t = t_HB[i+1]-t_HB[i]
        intervals_satisfy_nyquist.append(current_delta_t<delta_t_nyquist)
        if current_delta_t < delta_t_nyquist:
            intervals_satisfy_nyquist.append(True)
        else:
            nyquist_satisfied_for_all = False
            intervals_satisfy_nyquist.append(False)
    
    # create a string with information characterizing the time instances
    if len(t_HB) == 2*K+1:
        t_HB_info = '('+time_discretization+' - '+str(2*K+1)+' instances '
    else:        
        t_HB_info = '('+time_discretization+' instances '
    if nyquist_satisfied_for_all:
        t_HB_info += '-- nyquist_satisfied'
    t_HB_info += ')'
    
    
    
    # plot the eigenvalues of the HB operator matrix
    plot_eigenvalues(D_HB, auto_open=False)
    
    # again, pull out the exact function values at the time instances
    f_at_t_HB, dummy = my_non_periodic_fun(t_HB, actual_omegas)
    # multiply by the HB operator matrix 
    df_HB = np.dot(D_HB, f_at_t_HB)
    
    # set a fontsize for all axes and labels
    the_fontsize = 18    
    
    # plot everything
    plot_name = 'HB_operator_check '+t_HB_info
    auto_open = True
    trig_interp_HB = all_omegas_valid
    #trig_interp_HB = False
    trig_interp_HB = True
    plt.figure(plot_name)
    #plt.plot(t, f, label='$f_{exact}$')
    plt.plot(t, df, 'r-', label='$df/dt$')
    #plt.plot(t_HB, f_at_t_HB, 'ko', label='$f_{HB}$')
    if trig_interp_HB:
        plt.plot(t_HB, df_HB, 'go', label='$\\dot{\\mathbf{f}}$')
        t_HB_int, df_HB_int, dummy = trig_interp_given_freqs(t_HB,df_HB,omegas,t)
        plt.plot(t_HB_int,df_HB_int, 'g--', label='$\\dot{\\mathbf{f}}_{fine}$')
    else:
        plt.plot(t_HB, df_HB, 'go--', label='$df/dt_{HB}$')
        dummy = 1
    plt.xlabel('$t$', fontsize=the_fontsize)
    plt.ylabel('$\\frac{df}{dt}$', fontsize=the_fontsize)
    plt.legend(loc='best', fontsize=the_fontsize)
    plt.tight_layout()
    #plt.title('$\omega_{actual} = \{' + str(actual_omegas)[1:-1] + \
    #          '\} \quad\quad \omega_{used} = \{'+str(omegas)[1:-1]+'\}$', fontsize=12)
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
    # check to see if the HB expansion works #
    ##########################################
    
    # again, pull out the exact function values at the time instances
    f_at_t_HB, dummy = my_non_periodic_fun(t_HB, actual_omegas)
    # construct the the forward "transform" matrix
    G = HB_expansion_matrix(omegas, time_discretization=t_HB)
    
    #print('\nG =', G)
    
    # get the dimensions of G
    (m,n) = np.shape(G)
    # check to see if the G matrix is square
    G_is_square = m==n
    # check to see if G has full rank
    G_has_full_rank = my_rank(G)==min(m,n)
    
    print('\n\tnp.linalg.matrix_rank(G) =', np.linalg.matrix_rank(G))
    print('\n\tnp.linalg.cond(G) =', np.linalg.cond(G))
    print('\n\tmy_rank(G) =', my_rank(G))
    
    
    
    # see how many solution c are possible, using Rouche-Capelli
    N_c_solutions = rouche_capelli_check(G,f_at_t_HB, verbose=True)
    print('\tno. of possible c vectors: ', N_c_solutions, '\n\n')
    
    
    # compute HB expansion coefficients by taking the pseudoinverse of G
    try:
        c_pinv = np.dot(my_pinv(G), f_at_t_HB)
        using_approx_G_plus = False
    except ValueError:
        print('\n\t G matrix doesn\'t have full rank, using an approximate pseudoinverse...\n')
        c_pinv = np.dot(my_pinv(G, approximate_using_svd=True), f_at_t_HB)
        using_approx_G_plus = True
    print('\nc_pinv =', c_pinv)
    
    
    
    
    # *** PUT IN A SEPARATE FUNTION ***
    # try to predict what the erroneous coefficients will look like under T1
    if all_omegas_valid:
        print('\npredictions:')
        print('\n  CASE I.A:')
        for i in range(K-1,-1,-1):
            # just the real part of coeffs corresponding the affected frequency
            print('\n    -if omega_'+str(K-i)+' affected: c_tilde_-'+str(K-i)+' = c_tilde_'+str(K-i)+' = '+str((1/2)*(c_pinv[i]+c_pinv[-(i+1)]))[1:-1])
            Im_c_i = np.imag(c_pinv[-(i+1)])
            case_1A_error = np.sqrt(2.0)*abs(Im_c_i)
            print('    \t error: ||c-c_tilde||_2 = '+str(case_1A_error)[1:-1])
        
        print('\n  CASE I.B:')
        for i in range(K-1,-1,-1):
            # one third of the DC coeff plus twice the real part of coeffs 
            # corresponding the affected frequency
            print('\n    -if omega_'+str(K-i)+' affected: c_tilde_0 = c_tilde_-'+str(K-i)+' = c_tilde_'+str(K-i)+' = '+str((1/3)*(c_pinv[K]+c_pinv[i]+c_pinv[-(i+1)]))[1:-1])
            # DC is always real, just getting rid of the imaginary part here
            c_0 = np.real(c_pinv[K])
            Re_c_i = np.real(c_pinv[i])
            case_1B_error = np.sqrt(2.0/3.0)*np.sqrt(c_0**2.0 - 2.0*Re_c_i*(c_0-3.0*Re_c_i) - 3.0*abs(c_pinv[i])**2.0)
            print('    \t error: ||c-c_tilde||_2 = '+str(case_1B_error)[1:-1])

        print('\n  CASE II:')
        print('\n      pivot   copied\t\tc~_k" \t\t      error: ||c-c_tilde||_2')
        print('      -----   ------   ----------------------------   -----------------------')
        for pivot_col in range(1,K):
            pivot_col_index = pivot_col+K
            possible_copied_cols = list(range(pivot_col+1,K+1))
            possible_copied_columns = [-possible_copied_cols[-i-1] for i in range(len(possible_copied_cols))] + possible_copied_cols
            for copied_col in possible_copied_columns:
                copied_col_index = copied_col+K
                c_tilde_k_double_prime = 0.5*(c_pinv[pivot_col_index]+c_pinv[copied_col_index])
                case_2_error = np.sqrt(abs(c_pinv[pivot_col_index])**2 + \
                                abs(c_pinv[copied_col_index])**2 - \
                                2*(np.real(c_pinv[pivot_col_index])*np.real(c_pinv[copied_col_index]) + \
                                np.imag(c_pinv[pivot_col_index])*np.imag(c_pinv[copied_col_index])))
                
                if np.sign(copied_col) == 1:
                    print('\t'+str(pivot_col)+'\t '+str(copied_col)+'\t'+str(c_tilde_k_double_prime)[1:-1]+'\t\t   '+str(case_2_error)[1:-1])
                else:
                    print('\t'+str(pivot_col)+'\t'+str(copied_col)+'\t'+str(c_tilde_k_double_prime)[1:-1]+'\t\t   '+str(case_2_error)[1:-1])





    # reconstruct the function values from the coefficients
    f_expansion_at_t_HB_pinv = np.real(np.dot(G,c_pinv))
    #print('\nf_expansion_at_t_HB_pinv =',f_expansion_at_t_HB_pinv)
    
    # if you can, compute the HB expansion coefficients using the inverse too
    if G_is_square and G_has_full_rank:
        c_inv = np.dot(my_inv(G),f_at_t_HB)
        print('\nc_inv =', c_inv)
        # reconstruct the function values from the coefficients
        f_expansion_at_t_HB_inv = np.real(np.dot(G,c_inv))
        #print('\nf_expansion_at_t_HB_inv =',f_expansion_at_t_HB_inv)
        # c_inv should be the same as c_pinv here. make sure it is!
        if np.linalg.norm(c_pinv-c_inv) < 1e-10:
            print('\n\t c_inv = c_pinv \t ... \t because G is invertible: good.\n')
    
    # see if the exact noisy signal can be recovered using an 
    # interpolation scheme built up from the assumed frequencies
    # see how many solution c are possible, using Rouche-Capelli
    G_fine = HB_expansion_matrix(omegas, time_discretization=t)
    N_c_solutions = rouche_capelli_check(G_fine, f, verbose=True)
    print('\tno. of possible c vectors for the interpolation: ', N_c_solutions, '\n\n')
    
    # plot everything
    plot_name = 'HB_expansion_check '+t_HB_info
    auto_open = False
    if time_discretization == 'use_T1' and not all_omegas_valid:
        trig_interp_HB = False
    else:
        trig_interp_HB = True
    plt.figure(plot_name)
    plt.plot(t, f, 'k', label='$f_{exact}$')
    plt.plot(t_HB, f_at_t_HB, 'ko')    
    if G_is_square and G_has_full_rank:
        plt.plot(t_HB, f_expansion_at_t_HB_inv, 'r*', label='$f_{expansion, inv}$')
        if trig_interp_HB:
            t_HB_int, f_expansion_at_t_HB_inv_int, dummy = trig_interp_given_freqs(t_HB,f_expansion_at_t_HB_inv,omegas,t)
            plt.plot(t_HB_int,f_expansion_at_t_HB_inv_int, 'r--', label='$tonal\,\,interp.$')
        else:        
            plt.plot(t_HB, f_expansion_at_t_HB_inv, 'r*--')
    plt.plot(t_HB, f_expansion_at_t_HB_pinv, 'g*', label='$f_{expansion, pinv}$')
    t_HB_int, f_expansion_at_t_HB_pinv_int, dummy = trig_interp_given_freqs(t_HB,f_expansion_at_t_HB_pinv,omegas,t)
    plt.plot(t_HB_int,f_expansion_at_t_HB_pinv_int, 'g-.', label='$tonal\,\,interp.$')
    plt.xlabel('$t$', fontsize=14)
    plt.ylabel('$f(t)$', fontsize=14)
    plt.legend(loc='best')
    plt.title('$\omega_{actual} = \{' + str(actual_omegas)[1:-1] + \
              '\} \quad\quad \omega_{used} = \{'+str(omegas)[1:-1]+'\}$', fontsize=14)
    # save plot and close
    print('\n\t'+'saving final image...', end='')
    file_name = plot_name+'.png'
    plt.savefig(file_name, dpi=300)
    print('figure saved: '+plot_name)
    plt.close(plot_name)
    # open the saved image, if desired
    if auto_open:
        webbrowser.open(file_name)
    
    # if trig_interp_HB has been turned off, turn it back on and plot again
    trig_interp_HB = False
    if not trig_interp_HB:
        # plot everything
        plot_name = 'HB_expansion_check-with_trig_interp '+t_HB_info
        auto_open = True
        plt.figure(plot_name)
        plt.plot(t, f, 'k-', label='$f$')
        plt.plot(t_HB, f_at_t_HB, 'go', label='$\\mathbf{f}$')
        if G_is_square and G_has_full_rank:
            #plt.plot(t_HB, f_expansion_at_t_HB_inv, 'r*', label='$f_{expansion, inv}$')
            t_HB_int, f_expansion_at_t_HB_inv_int, dummy = trig_interp_given_freqs(t_HB,f_expansion_at_t_HB_inv,omegas,t)
            #plt.plot(t_HB_int,f_expansion_at_t_HB_inv_int, 'r--', label='$tonal\,\,interp.$')
        #plt.plot(t_HB, f_expansion_at_t_HB_pinv, 'g*', label='$f_{expansion, pinv}$')
        t_HB_int, f_expansion_at_t_HB_pinv_int, dummy = trig_interp_given_freqs(t_HB,f_expansion_at_t_HB_pinv,omegas,t)
        plt.plot(t_HB_int,f_expansion_at_t_HB_pinv_int, 'g-.', label='$\\mathbf{f}_{fine}$')
        plt.xlabel('$t$', fontsize=the_fontsize)
        plt.ylabel('$f(t)$', fontsize=the_fontsize)
        plt.legend(loc='best', fontsize=the_fontsize)
        plt.tight_layout()
        #plt.title('$\omega_{actual} = \{' + str(actual_omegas)[1:-1] + \
        #          '\} \quad\quad \omega_{used} = \{'+str(omegas)[1:-1]+'\}$', fontsize=14)
        # save plot and close
        print('\n\t'+'saving final image...', end='')
        file_name = plot_name+'.png'
        plt.savefig(file_name, dpi=300)
        print('figure saved: '+plot_name)
        plt.close(plot_name)
        # open the saved image, if desired
        if auto_open:
            webbrowser.open(file_name)
    
    
    # compute the sensitivity of the expansion and the derivative to noise
    print('\n\tevaluating noise percentages', end='')
    percent_noises = np.linspace(0.0, 40.0)
    trials_per_percentage = 50
    average_expansion_residuals = []
    average_fine_expansion_residuals = []
    average_derivative_residuals = []
    average_fine_derivative_residuals = []
    # HB interpolation matrix (HB expansion matrix built w/the fine time grid)
    G_fine = HB_expansion_matrix(omegas, time_discretization=t)
    # diagonal frequency matrix (used while computing the fine-grid derivative)
    w = [-omegas[-(i+1)] for i in range(len(omegas))]+[0.0]+omegas
    w_imag = [1j*w_i for w_i in w]
    Omega = np.diag(w_imag)

    # specify which norm to use for the residual computation (p=1,2,...,np.inf) **************
    p = 2
    # plot on log-log axes?
    use_loglog_axes = False
    
    for noise_percentage in percent_noises:
        if noise_percentage/max(percent_noises) % 0.1 < 1e-2:
            print('.', end='')
        expansion_residuals = []
        derivative_residuals = []
        fine_expansion_residuals = []
        fine_derivative_residuals = []
        for _ in range(trials_per_percentage):
            
            # create a set of omegas that's a noisy version of the actual omegas
            # let noise be <(noise_percentage)% of the actual values
            noises = (noise_percentage/100.0)*np.array(actual_omegas)*(np.random.rand(len(actual_omegas))-0.5)
            noisy_omegas = list(np.array(actual_omegas) + noises)
            
            # "exact" / "fine-grid" function containing only these noisy omegas
            f_noisy, df_noisy = my_non_periodic_fun(t, noisy_omegas)
            
            # pull out the exact noisy-function values at the time instances
            f_noisy_at_t_HB, df_noisy_at_t_HB = my_non_periodic_fun(t_HB, noisy_omegas)
            
            # compute HB expansion coefficients by taking the pseudoinverse of G
            try:
                c_pinv = np.dot(my_pinv(G), f_noisy_at_t_HB)
                using_approx_G_plus = False
            except ValueError:
                #print('\n\t G matrix doesn\'t have full rank, using an approximate pseudoinverse...\n')
                c_pinv = np.dot(my_pinv(G, approximate_using_svd=True), f_noisy_at_t_HB)
                using_approx_G_plus = True
            
            # reconstruct the function values at t_HB from the coefficients
            f_noisy_expansion_at_t_HB_pinv = np.real(np.dot(G,c_pinv))
            
            # if you can, compute the HB expansion coefficients using the inverse
            if G_is_square and G_has_full_rank:
                c_inv = np.dot(my_inv(G),f_noisy_at_t_HB)
                f_noisy_expansion_at_t_HB_inv = np.real(np.dot(G,c_inv))            
            
            # compute a measure of the discrepancy between the samples and the
            # reconstruction at the time instances
            expansion_residual = f_noisy_expansion_at_t_HB_pinv - f_noisy_at_t_HB
            norm_expansion_residual = np.linalg.norm(expansion_residual.squeeze(),ord=p)
            expansion_residuals.append(norm_expansion_residual)
            
            # try reconstructing the function on the fine grid from the coeffs
            # (yes, this will be the same as the above line at the t_HB)
            f_noisy_expansion_fine = np.real(np.dot(G_fine,c_pinv))
            
            # compute a measure of the discrepancy between the fine-grid
            # reconstruction and the exact noisy signal
            fine_expansion_residual = f_noisy_expansion_fine - f_noisy
            norm_fine_expansion_residual = np.linalg.norm(fine_expansion_residual.squeeze(),ord=p)
            fine_expansion_residuals.append(norm_fine_expansion_residual)
            
            # compute the derivative at the time instances using the HB operator
            df_noisy_HB = np.dot(D_HB, f_noisy_at_t_HB)
            df_noisy_HB = np.real(np.dot(G,np.dot(Omega, c_pinv)))
            
            # compute a measure of the discrepancy between the sampled 
            # derivatives and the HB derivatives at the time instances
            derivative_residual = df_noisy_HB - df_noisy_at_t_HB
            norm_derivative_residual = np.linalg.norm(derivative_residual.squeeze(),ord=p)
            derivative_residuals.append(norm_derivative_residual)
            
            # try reconstructing the derivative on the fine grid from the
            # coeffs
            df_noisy_fine = np.real(np.dot(G_fine,np.dot(Omega, c_pinv)))
            
            # compute a measure of the discrepancy between the fine-grid
            # derivative and the exact noisy derivative
            fine_derivative_residual = df_noisy_fine - df_noisy
            norm_fine_derivative_residual = np.linalg.norm(fine_derivative_residual.squeeze(),ord=p)
            fine_derivative_residuals.append(norm_fine_derivative_residual)
            
        # compute & store the average residual norms found for this noise level
        average_expansion_residual = sum(expansion_residuals)/trials_per_percentage
        average_expansion_residuals.append(average_expansion_residual)
        average_fine_expansion_residual = sum(fine_expansion_residuals)/trials_per_percentage
        average_fine_expansion_residuals.append(average_fine_expansion_residual)
        average_derivative_residual = sum(derivative_residuals)/trials_per_percentage
        average_derivative_residuals.append(average_derivative_residual)
        average_fine_derivative_residual = sum(fine_derivative_residuals)/trials_per_percentage
        average_fine_derivative_residuals.append(average_fine_derivative_residual)
    # close off the monitor
    print('done!\n')
    
    # at the last trial for the last noise level, check rouche-capelli
    print('\n\n - for the '+str(trials_per_percentage)+'-th trial with noise of: ', round(noise_percentage,3), '%:')
    # see if the noisy signal, sampled at the HB points, can be expanded in the
    # columns of the HB expansion matrix, using Rouche-Capelli
    print('\n  - for the extrapolation coefficients:')
    N_c_solutions = rouche_capelli_check(G, f_noisy_at_t_HB, verbose=True)
    print('\t    no. of exact c vectors for the HB expansion: ', N_c_solutions)
    
    # see if the exact noisy signal can be recovered using an 
    # interpolation scheme built up from the assumed frequencies
    # see how many solution c are possible, using Rouche-Capelli
    print('\n  - for the interpolation coefficients:')
    G_fine = HB_expansion_matrix(omegas, time_discretization=t)
    N_c_solutions = rouche_capelli_check(G_fine, f_noisy, verbose=True)
    print('\t    no. of exact c vectors for the interpolation: ', N_c_solutions, '\n\n')
    
    # plot sensitivity of the expansion and the derivative to noisy omegas
    plot_name = 'HB_noise_sensitivity '+t_HB_info
    auto_open = False
    plt.figure(plot_name)
    if using_approx_G_plus:
        expansion_label = '$f|_{HB}-G\\tilde{G}^+ f_{HB}$'
        fine_expansion_label = '$f-G_{fine}\\tilde{G}^+ f_{HB}$'
        derivative_label = '$\\frac{\partial f}{\partial t}|_{HB}-G \Omega \\tilde{G}^+ f_{HB}$'
        fine_derivative_label = '$\\frac{\partial f}{\partial t}-G_{fine} \Omega \\tilde{G}^+ f_{HB}$'
    else:
        expansion_label = '$f|_{HB}-GG^+ f_{HB}$'
        fine_expansion_label = '$f-G_{fine}G^+ f_{HB}$'
        derivative_label = '$\\frac{\partial f}{\partial t}|_{HB}-G \Omega G^+ f_{HB}$'
        fine_derivative_label = '$\\frac{\partial f}{\partial t}-G_{fine} \Omega G^+ f_{HB}$'
    if use_loglog_axes:
        plt.loglog(percent_noises, average_expansion_residuals, 'b.-', label=expansion_label)
        plt.loglog(percent_noises, average_fine_expansion_residuals, 'c.-', label=fine_expansion_label)
        plt.loglog(percent_noises, average_derivative_residuals, 'r.-', label=derivative_label)
        plt.loglog(percent_noises, average_fine_derivative_residuals, 'm.-', label=fine_derivative_label)
    else:
        plt.semilogy(percent_noises, average_expansion_residuals, 'b.-', label=expansion_label)
        plt.semilogy(percent_noises, average_fine_expansion_residuals, 'c.-', label=fine_expansion_label)
        plt.semilogy(percent_noises, average_derivative_residuals, 'r.-', label=derivative_label)
        plt.semilogy(percent_noises, average_fine_derivative_residuals, 'm.-', label=fine_derivative_label)
    
    plt.xlabel('$\% \,noise \; present \; in \; each \; actual \; \omega_i$', fontsize=14)
    if p == np.inf:
        p_string = '\infty'
    else:
        p_string = str(p)
    plt.ylabel('$\overline{\|residual\|_{'+p_string+'}}$', fontsize=14)
    plt.ylim(1e-14,1e5)
    plt.legend(loc='best')
    plt.title('$\omega_{used} = \{'+str(omegas)[1:-1]+'\}$', fontsize=14)
    # save plot and close
    print('\n\t'+'saving final image...', end='')
    file_name = plot_name+'.png'
    plt.savefig(file_name, dpi=300)
    print('figure saved: '+plot_name)
    plt.close(plot_name)
    # open the saved image, if desired
    if auto_open:
        webbrowser.open(file_name)
        
    # plot everything
    plot_name = 'noisy_HB_expansion_check '+t_HB_info
    auto_open = False
    if time_discretization == 'use_T1' and not all_omegas_valid:
        trig_interp_HB = False
    else:
        trig_interp_HB = True
    plt.figure(plot_name)
    plt.plot(t, f_noisy, 'k', label='$f_{exact}$')
    plt.plot(t_HB, f_noisy_at_t_HB, 'ko')    
    if G_is_square and G_has_full_rank:
        plt.plot(t_HB, f_noisy_expansion_at_t_HB_inv, 'r*', label='$f_{expansion,inv}$')
        if trig_interp_HB:
            #t_HB_int, f_noisy_expansion_at_t_HB_inv_int, dummy = trig_interp_given_freqs(t_HB,f_noisy_expansion_at_t_HB_inv, omegas,t)
            #plt.plot(t_HB_int,f_noisy_expansion_at_t_HB_inv_int, 'r--', label='$tonal\,\,interp.$')
            plt.plot(t,f_noisy_expansion_fine, 'r--', label='$tonal\,\,interp.$')
        else:
            plt.plot(t_HB, f_noisy_expansion_at_t_HB_inv, 'r*--')
    if trig_interp_HB:
        plt.plot(t_HB, f_noisy_expansion_at_t_HB_pinv, 'g*', label='$f_{expansion,pinv}$')
        #t_HB_int, f_noisy_expansion_at_t_HB_pinv_int, dummy = trig_interp_given_freqs(t_HB,f_noisy_expansion_at_t_HB_pinv, omegas,t)
        #plt.plot(t_HB_int,f_noisy_expansion_at_t_HB_pinv_int, 'g-.', label='$tonal\,\,interp.$')
        plt.plot(t,f_noisy_expansion_fine, 'g-.', label='$tonal\,\,interp.$')
    else:
        plt.plot(t_HB, f_noisy_expansion_at_t_HB_pinv, 'g*--', label='$f_{expansion,pinv}$')
        
    plt.xlabel('$t$', fontsize=14)
    plt.ylabel('$f(t)$', fontsize=14)
    plt.legend(loc='best')
    rounded_noisy_omegas = [float('{0:.4g}'.format(omega)) for omega in noisy_omegas]
    plt.title('$\omega_{actual} = \{' + str(rounded_noisy_omegas)[1:-1] + \
              '\} \quad \omega_{used} = \{'+str(omegas)[1:-1]+'\}$', fontsize=14)
    # save plot and close
    print('\n\t'+'saving final image...', end='')
    file_name = plot_name+'.png'
    plt.savefig(file_name, dpi=300)
    print('figure saved: '+plot_name)
    plt.close(plot_name)
    # open the saved image, if desired
    if auto_open:
        webbrowser.open(file_name)
    
    # if trig_interp_HB has been turned off, turn it back on and plot again
    if not trig_interp_HB:
        # plot everything
        plot_name = 'HB_expansion_check + noise-with_trig_interp '+t_HB_info
        auto_open = True
        plt.figure(plot_name)
        plt.plot(t, f_noisy, 'k', label='$f_{exact}$')
        plt.plot(t_HB, f_noisy_at_t_HB, 'ko')
        if G_is_square and G_has_full_rank:
            plt.plot(t_HB, f_noisy_expansion_at_t_HB_inv, 'r*', label='$f_{expansion,inv}$')
            t_HB_int, f_noisy_expansion_at_t_HB_inv_int, dummy = trig_interp_given_freqs(t_HB,f_noisy_expansion_at_t_HB_inv, omegas,t)
            plt.plot(t_HB_int,f_noisy_expansion_at_t_HB_inv_int, 'r--', label='$tonal\,\,interp.$')        
        plt.plot(t_HB, f_noisy_expansion_at_t_HB_pinv, 'g*', label='$f_{expansion,pinv}$')
        t_HB_int, f_noisy_expansion_at_t_HB_pinv_int, dummy = trig_interp_given_freqs(t_HB,f_noisy_expansion_at_t_HB_pinv, omegas,t)
        plt.plot(t_HB_int,f_noisy_expansion_at_t_HB_pinv_int, 'g-.', label='$tonal\,\,interp.$')
        plt.xlabel('$t$', fontsize=14)
        plt.ylabel('$f(t)$', fontsize=14)
        plt.legend(loc='best')
        rounded_noisy_omegas = [float('{0:.4g}'.format(omega)) for omega in noisy_omegas]
        plt.title('$\omega_{actual} = \{' + str(rounded_noisy_omegas)[1:-1] + \
                  '\} \quad \omega_{used} = \{'+str(omegas)[1:-1]+'\}$', fontsize=14)
        # save plot and close
        print('\n\t'+'saving final image...', end='')
        file_name = plot_name+'.png'
        plt.savefig(file_name, dpi=300)
        print('figure saved: '+plot_name)
        plt.close(plot_name)
        # open the saved image, if desired
        if auto_open:
            webbrowser.open(file_name)
    
    # noisy derivative check
    plot_name = 'noisy_HB_operator_check '+t_HB_info
    auto_open = False
    trig_interp_HB = all_omegas_valid    
    plt.figure(plot_name)
    plt.plot(t, f_noisy, label='$f_{exact}$')
    plt.plot(t, df_noisy, 'r-', label='$df/dt_{exact}$')
    plt.plot(t_HB, f_noisy_at_t_HB, 'ko', label='$f_{HB}$')
    if trig_interp_HB:
        plt.plot(t_HB, df_noisy_HB, 'go', label='$df/dt_{HB}$')
        #t_HB_int, df_noisy_HB_int, dummy = trig_interp_given_freqs(t_HB,df_noisy_HB,omegas,t)
        #plt.plot(t_HB_int,df_noisy_HB_int, 'g--', label='$tonal\,\,interp.$')
        plt.plot(t,df_noisy_fine, 'g--', label='$tonal\,\,interp.$')
    else:
        plt.plot(t_HB, df_noisy_HB, 'go--', label='$df/dt_{HB}$')
        dummy = 1
    plt.xlabel('$t$', fontsize=16)
    plt.ylabel('$f(t)$', fontsize=16)
    plt.legend(loc='best')
    rounded_noisy_omegas = [float('{0:.4g}'.format(omega)) for omega in noisy_omegas]
    plt.title('$\omega_{actual} = \{' + str(rounded_noisy_omegas)[1:-1] + \
                  '\} \quad \omega_{used} = \{'+str(omegas)[1:-1]+'\}$', fontsize=14)
    # save plot and close
    print('\n\t'+'saving final image...', end='')
    file_name = plot_name+'.png'
    plt.savefig(file_name, dpi=300)
    print('figure saved: '+plot_name)
    plt.close(plot_name)
    # open the saved image, if desired
    if auto_open:
        webbrowser.open(file_name)
    
    # if trig_interp_HB has been turned off, turn it back on and plot again
    # remember, "fourier interp" is just evaluation on a fine grid
    if not trig_interp_HB:
        # replot the noisy operator check, but with Fourier interpolation
        plot_name = 'noisy_HB_operator_check -with_trig_interp '+t_HB_info
        auto_open = False
        plt.figure(plot_name)
        plt.plot(t, f_noisy, label='$f_{exact}$')
        plt.plot(t, df_noisy, 'r-', label='$df/dt_{exact}$')
        plt.plot(t_HB, f_noisy_at_t_HB, 'ko', label='$f_{HB}$')
        plt.plot(t_HB, df_noisy_HB, 'go', label='$df/dt_{HB}$')
        plt.plot(t,df_noisy_fine, 'g--', label='$tonal\,\,interp.$')
        plt.xlabel('$t$', fontsize=16)
        plt.ylabel('$f(t)$', fontsize=16)
        plt.legend(loc='best')
        rounded_noisy_omegas = [float('{0:.4g}'.format(omega)) for omega in noisy_omegas]
        plt.title('$\omega_{actual} = \{' + str(rounded_noisy_omegas)[1:-1] + \
                      '\} \quad \omega_{used} = \{'+str(omegas)[1:-1]+'\}$', fontsize=14)
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
    run_this_section = False            # do you want to run this analysis?
    # if desired, perform this study
    if run_this_section:
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
        t_HB_int_actual, df_HB_int_actual, dummy = trig_interp_given_freqs(t_HB, df_HB, actual_omegas)
        # loop through the various errors to try
        for error_percentage in percent_errors:
            # set the wrong angular frequencies
            trial_omegas = [(1.0+error_percentage/100)*omega for omega in actual_omegas]
            # find the interpolated HB derivative with the incorrect frequencies
            D_HB, t_HB = harmonic_balance_operator(trial_omegas, time_discretization)
            f_HB, dummy = my_non_periodic_fun(t_HB, actual_omegas)
            df_HB = np.dot(D_HB, f_HB)
            t_HB_int_wrong, df_HB_int_wrong, dummy = trig_interp_given_freqs(t_HB,df_HB,trial_omegas)
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
        t_HB_int, df_HB_int_wrong, dummy = trig_interp_given_freqs(t_HB,df_HB,max_omegas)
        
        # plot the correct derivative and the worst answer studied
        plot_name = 'result_at_worst_answer'
        auto_open = False
        plt.figure(plot_name)
        plt.plot(t, f, label='$f_{exact}$')
        plt.plot(t_HB_actual, f_HB_actual, 'ko', label='$f_{HB}$')
        plt.plot(t, df, 'r-', label='$df/dt_{exact}$')
        plt.plot(t_HB, df_HB_wrong, 'go', label='$df/dt_{HB,wrong}$')
        t_HB_int, df_HB_int, dummy = trig_interp_given_freqs(t_HB,df_HB_wrong,max_omegas)
        plt.plot(t_HB_int,df_HB_int, 'g--', label='$tonal\,\,interp.$')
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
    # See how cond(D_HB) and cond(G) vary with selected angular frequencies #
    #############################################################################
    first_omega = omegas[0]
    starting_multiple = 1
    penultimate_omega = omegas[-2]
    penultimate_multiple = penultimate_omega/first_omega
    nondim_penultimate_omega = 2.0*(penultimate_omega-first_omega)/(penultimate_omega+first_omega)
    max_multiple = 25.0
    multiples = myLinspace(starting_multiple, max_multiple, 100*int(max_multiple-starting_multiple)+1)
    plot_nondim_omegas = False
    plot_outliers_in_title = False
    fully_periodic_signal = False    # omegas includes fundamental and consecutive multiples
    
    # print message to the screen
    print('\nstudying the effect of different frequencies pairs on cond(D_HB) and cond(G)...')
    D_conds = []
    G_conds = []
    average_errors = []
    nondim_omegas = []
    for multiple in multiples:
        # omegas
        final_omega = first_omega*multiple
        freqs = [first_omega]+omegas[1:-1]+[final_omega]
        print('\n----')
        print('freqs =', freqs)
        nondim_omegas.append(2.0*(final_omega-first_omega)/(final_omega+first_omega))
        # operator
        D_HB, t_HB = harmonic_balance_operator(freqs, time_discretization)
        current_D_cond = np.linalg.cond(D_HB)
        D_conds.append(current_D_cond)
        # E inverse
        G = HB_expansion_matrix(freqs, time_discretization)
        current_G_cond = np.linalg.cond(G)
        G_conds.append(current_G_cond)
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
    G_cutoff = 1e12
    G_peaks = [float("{0:.3f}".format(multiples[i])) for i in range(len(G_conds)) if G_conds[i] > G_cutoff]
    G_nondim_peaks = [float("{0:.3f}".format(nondim_omegas[i])) for i in range(len(G_conds)) if G_conds[i] > G_cutoff]
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
        if plot_outliers_in_title:
            plt.title('$outliers \,\, at: \,\, '+str(D_nondim_peaks)[1:-1]+'$', fontsize=8)
    else:
        plt.semilogy(multiples, D_conds, 'k.-')
        plt.semilogy([penultimate_multiple]*2,[min(D_conds),max(D_conds)],'y--')
        plt.xlabel('$\omega_K/\omega_1$', fontsize=14)
        if plot_outliers_in_title:
            plt.title('$outliers \,\, at: \,\, '+str(D_peaks)[1:-1]+'$', fontsize=8)
    plt.ylabel('$\kappa(D_{HB})$', fontsize=16)
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
        
    # plot the G result
    plot_name = 'cond(G)_vs_freq_ratio'
    auto_open = False
    plt.figure(plot_name)
    if plot_nondim_omegas:
        plt.semilogy(nondim_omegas, G_conds, 'k.-')
        plt.semilogy([nondim_penultimate_omega]*2,[min(G_conds),max(G_conds)],'y--')
        plt.xlabel('$\delta^*_{\omega_K}=2\,\\frac{\omega_K-\omega_1}{\omega_K+\omega_1}$', fontsize=the_fontsize)
        if plot_outliers_in_title:
            plt.title('$outliers \,\, at: \,\, '+str(G_nondim_peaks)[1:-1]+'$', fontsize=6)
        plt.ylim(1,100)
        #plt.ylim(1,10000)
    else:
        plt.semilogy(multiples, G_conds, 'k.-')
        plt.semilogy([penultimate_multiple]*2,[min(G_conds),max(G_conds)],'y--')
        if fully_periodic_signal:
            cond_at_K = np.interp(K,multiples, G_conds)
            plt.semilogy(K, cond_at_K,'co', label='$fully\; periodic$')
            plt.legend(loc='best')
        plt.xlabel('$\omega_{'+str(K)+'}/\omega_1$', fontsize=the_fontsize)
        plt.xlabel('$\omega_K/\omega_1$', fontsize=the_fontsize)
        if plot_outliers_in_title:
            plt.title('$outliers \,\, at: \,\, '+str(G_peaks)[1:-1]+'$', fontsize=6)
    plt.ylabel('$\kappa\\left(\\mathbf{G}\\right)$', fontsize=the_fontsize)
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
        plt.semilogy([penultimate_multiple]*2,[min(average_errors),max(average_errors)],'y--')
        if fully_periodic_signal:
            plt.semilogy([K]*2,[min(average_errors),max(average_errors)],'c--')        
        plt.xlabel('$\omega_{'+str(K)+'}/\omega_1$', fontsize=the_fontsize)
        #plt.xlabel('$\omega_K/\omega_1$', fontsize=the_fontsize)
        if plot_outliers_in_title:
            plt.title('$outliers \,\, at: \,\, '+str(df_error_peaks)[1:-1]+'$', fontsize=7)
            plt.semilogy(multiples,[df_percent_tol]*len(multiples),'r--')
    plt.ylabel('$\% \,\, error \,\, in \,\,\\dot{\\mathbf{f}}$', fontsize=14)
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
    
    t_start = 0.0                           # initial time
    t_end = 1.2*period_given_freqs(omegas)  # approx. final time (stop at or just after)
    
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
            f.append(f[n-1] + delta_t*the_ode.evaluate(times[n-1],f[n-1]))
    
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
    
    ########################################################
    # solve the equation using the harmonic-balance method #
    ########################################################
    
    # set the value to use "full convergence," using machine zero
    machine_zero = sys.float_info.epsilon
    fully_converged = 1e3*machine_zero
    
    # call the function to solve the HB problem
    t_HB, f_HB = solve_HB_problem(omegas, time_discretization, the_ode, 
                                delta_tau=0.01, 
                                constant_init_guess=10.0, 
                                residual_convergence_criteria=fully_converged, 
                                make_plot=True, auto_open_plot=False, 
                                make_movie=False, auto_play_movie=False,
                                optimize_omegas=False)
                            
    # interpolate the harmonic-balance solution using the prescribed frequencies
    t_HB_int, f_HB_int, dummy = trig_interp_given_freqs(t_HB, f_HB, omegas)
    
    ###############################################################################
    # recursive aliasing check. starting with a guess of the largest frequency in 
    # the signal, keep doubling the bandwidth (B) and the number of uniformly 
    # spaced, positive angular frequencies to use (kappa) (initally start with 20) 
    # until the number and location of the "refined" peaks in the DFT converge 
    # (within prescribed error bands)
    ###############################################################################
    
    # [user input] intial guess for the largest tonal frequency 
    max_tone_guess = 9.5
    # initial number of uniformly spaced points to discretize the initial bandwidth
    initial_points_in_B = 25
    # set level of "partial convergence" for the trial problems
    partial_convergence = 1e-1
    # set the maximum number of bandwidth contractions to try
    max_contractions = 7
    # minimum percentage AC energy to be captured in the extracted peaks
    AC_energy_to_capture = 95
    # give a constant initial guess of for the HB solution
    HB_initial_guess = 10.0
    # time discretization for defining HB time instances ('use_T1', 'use_Nyquist')
    time_discretization = 'use_T1'
    
    # print message to the screen
    print('\nstarting aliasing protection process...\n')
    print('\n\t-----------------------')
    print('\n\t   inital bandwidth')
    print('\n\t-----------------------\n')
    # set the initial bandwidth to the guess for the largest tonal peak
    B = max_tone_guess
    # set the number and location of the initial set of frequencies to the desired
    # number of initial points rounded to the nearest power of two (to use with the 
    # Cooley and Tukey FFT algorithm)
    kappa = int(2.0**np.round(np.log2(initial_points_in_B)))
    # discretize B, partially solve HB problem, take the DFT, return peak details
    n_peaks, peaks_found, \
    peak_ranges, peak_boundaries, ave_f_HB = HB_sol_plus_DFT(B, kappa,
                                                              time_discretization, 
                                                              the_ode,
                                                              HB_initial_guess, 
                                                              partial_convergence,
                                                              AC_energy_to_capture) 
    # extend the bandwidth until the extracted peaks have converged within bounds
    peaks_converged = False
    for i in range(max_contractions):
        # print a header for this iteration
        print('\n\t------------------------')
        print('\n\tbandwidth contraction #'+str(i+1))
        print('\n\t------------------------\n')
        # reassign the number and locations of the peaks from the previous run
        n_peaks_old = n_peaks
        peak_ranges_old = copy.copy(peak_ranges)
        # set the new bandwidth equal to the largest clustered peak's upper 
        # boundary and now use about twice the number of points within this range
        # (rounded to the nearest power of two...for use with the FFT algorithm)
        max_peak_upper_bound = peak_boundaries[-1][-1]
        B = max_peak_upper_bound
        twice_scaled_kappa = 2.0*(max_peak_upper_bound/B)*kappa
        kappa = int(2.0**np.round(np.log2(twice_scaled_kappa)))    
        # discretize B, partially solve HB problem, take the DFT, return peak details
        n_peaks, peaks_found, \
        peak_ranges, peak_boundaries, ave_f_FB = HB_sol_plus_DFT(B, kappa, 
                                                                time_discretization, 
                                                                the_ode, 
                                                                ave_f_HB,
                                                                partial_convergence, 
                                                                AC_energy_to_capture) 
        # check to see if the same number of peaks have been found as the last run
        if n_peaks == n_peaks_old:
            # compare the new peak ranges with the old ones
            for j in range(n_peaks):
                # extract the bounds of the old range for this peak
                lower_old = peak_ranges_old[j][0]
                upper_old = peak_ranges_old[j][1]
                # extract the bounds of the new range for this peak
                lower_new = peak_ranges[j][0]
                upper_new = peak_ranges[j][1]
                # check to see if the ranges are overlapping
                if lower_old <= upper_new and lower_new <= upper_old:
                    # if they are, then we've found a converged peak
                    peaks_converged = True
                else:
                    # we've found a non-converged peak and need to extend again
                    peaks_converged = False
                    break    
        # if we've found converged peaks, then stop extending the bandwidth
        if peaks_converged:
            print('\n\t----------------------')
            print('\n\tconverged peaks found!')
            print('\n\t----------------------\n')
            break
    
    ######################################################################
    # use gradient descent and time-marching to optimize the frequencies #
    ######################################################################
    
    # use the converged peaks found from the bandwidth-contraction process
    omegas = peaks_found
    
    # FOR TESTING ONLY!!!
    omegas = copy.copy(actual_omegas)
    omegas[0] = 1.31
    omegas[0] = 1.6
    
    # "random" values
    #omegas = [2.4, 3.5, 4.6, 6.7]
    # "close enough" values
    #omegas = [1.4000, 2.0, 3.7, 4.7]
     
    # record the initial guess for the angular frequencies
    initial_guess_omegas = copy.copy(omegas)
    # using these omegas as initial guesses, solve an HB problem while optimizing
    # the angular frequencies using time-accurate comparisons and gradient descent
    # N.B. when optimizing frequencies, the function returns three variables
    t_HB, f_HB = solve_HB_problem(omegas, time_discretization, the_ode, 
                                    delta_tau=0.01, 
                                    constant_init_guess=10.0, 
                                    residual_convergence_criteria=fully_converged,
                                    make_plot=True, auto_open_plot=True, 
                                    make_movie=True, auto_play_movie=False,
                                    optimize_omegas=False)
        
    # interpolate the harmonic-balance solution using the prescribed frequencies
    t_HB_int, f_HB_int, dummy = trig_interp_given_freqs(t_HB, f_HB, omegas)
    
    # compute the "long period" corresponding the final set of omegas
    T_HB_sol = period_given_freqs(omegas)
    # now, round the angular-frequencies (to a desired number of decimal points) 
    # and recompute the 
    desired_decimals = 1
    rounded_omegas = [float(round(omega,desired_decimals)) for omega in omegas]
    T_HB_sol_rounded = period_given_freqs(rounded_omegas)
    
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
    span_HB = t_HB[-1]
    n_delta_t_per_span_HB = math.floor(span_HB/delta_t)
    span_HB_check = n_delta_t_per_span_HB*delta_t
    n_checkpoints = n_delta_t_per_span_HB+1
    t_HB_check = myLinspace(0.0, span_HB_check, n_checkpoints)
    t_HB_check,f_HB_check = linearInterp(t_HB_int, f_HB_int, t_HB_check, verbose=True)
    
    # figure out where along the "long" period of the time-accurate solution the 
    # harmonic-balance solution lies
    min_norm_diff = 1e6             # intitalized to a high value
    norm_diffs = []
    min_index = 0
    t_HB_int_min_shifted = []       # interpolated HB sol at the best minimum
    t_HB_min_shifted = []           # HB sol at the time instances at best minimum
    t_HB_int_trial_shifted = []     # "proving" (sliding) interpolated HB solution
    for i in range(-time_points,-n_checkpoints):
        current_f_TA_range = f[i:i+n_checkpoints]
        t_HB_int_current = [t+times[i] for t in t_HB_check]
        diff = [abs(f_HB_check[j]-current_f_TA_range[j]) for j in range(n_checkpoints)]
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
    for omega in rounded_omegas:
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
    
    # plot a close up of the HB solution and the time-accurate portion it matches
    print('generating close-up plot of the comparison...')
    # plotting preliminaries
    plot_name = 'comparison_HB_with_TA'
    auto_open = False
    plt.figure(plot_name)
    # plot the two curves
    plt.plot(t_HB_min_shifted[-1],f_HB,'go')
    plt.plot(t_HB_int_min_shifted[-1],f_HB_check,'g-', label='$harmonic \,\, balance$')
    plt.plot(times[min_index:min_index+n_checkpoints+1], f[min_index:min_index+n_checkpoints+1],'b-',label='$time \,\, accurate$')
    # plot labels, axes, title, legend
    plt.xlabel('$t$', fontsize=18)
    plt.ylabel('$f(t)$', fontsize=18)
    plt.xlim(t_HB_min_shifted[-1][0],t_HB_min_shifted[-1][0]+span_HB_check)
    #plt.ylim(min(f)-vertical_padding,max(f)+vertical_padding)
    plt.legend(loc='best')
    # save plot and close
    print('\n\t'+'saving final image...', end='')
    file_name = plot_name+'.png'
    plt.savefig(file_name, dpi=300)
    print('figure saved: '+plot_name)
    plt.close(plot_name)
    # open the saved image, if desired
    if auto_open:
        webbrowser.open(file_name)
    
    ################################################
    # print all intermediate results to the screen #
    ################################################
    
    # actual periods of the HB solution and the actual ODE solution
    print('\n------------------------------------------------------------------\n')
    print('\n-true periods of solutions, based on given angular frequencies:')
    print('\n\t(n.b. these values are not unique! different combinations of ')
    print('\tangular frequencies can yield the same "long" period)')
    print('\n\tperiod of the HB solution, based on omegas used:', round(T_HB_sol,3))
    print('\n\t( period of HB solution, based on rounded omegas:', round(T_HB_sol_rounded,3),')')
    print('\n\tperiod of the ODE solution:', round(T_actual_sol,3),'\n')
    
    # print information about the angular frequencies
    print('\n-a look at the angular frequencies:')
    print('\n\t-initial guess [rad/s]:')
    print('\n\t\t'+str(initial_guess_omegas))
    print('\n\t-values used [rad/s]:')
    print('\n\t\t'+str(omegas))
    print('\n\t-values used (rounded) [rad/s]:')
    print('\n\t\t'+str(rounded_omegas))
    print('\n\t-actual values [rad/s]:')
    print('\n\t\t'+str(actual_omegas))
    
    # print information about the HB operator
    print('\n\n-a look at the HB operator:')
    print('\n\tK = '+str(len(omegas)))
    N = 2*len(omegas)+1
    print('\n\tN = '+str(N))
    print('\n\tD_HB = ', str(np.around(D_HB,3))[1:-1].replace('\n','\n\t'+' '*7),'\n')
    print('\tdet(D_HB) =', np.linalg.det(D_HB),'\n')
    print('\tcond(D_HB) =', np.linalg.cond(D_HB),'\n')
    print('\trank(D_HB) = ',np.linalg.matrix_rank(D_HB),'\n')
    G = HB_expansion_matrix(omegas, time_discretization)
    N_G = [[0]*N for i in range(N)]
    N_G = np.zeros((N,N),dtype=np.complex_)
    for i in range(N):
        for j in range(N):
            N_G[i][j] = np.round(N*G[i][j],2)
    stringed_G = []
    for i in range(N):
        realed_row = [np.real_if_close(N_G[i][j],1000) for j in range(N)]        
        stringed_row = ''.join([char for char in str(realed_row) if char not in 'array()'])
        stringed_G.append(stringed_row)
    print('\n\tG = (1/'+str(N)+')'+str(stringed_G)[2:-2].replace("', '",'\n\t'+' '*13).replace(', ','\t'))
    print('\n\trank(G) = ',np.linalg.matrix_rank(G))
    
    # print both TA and HB average function values to the screen
    print('\n-comparing the average value of the time-accurate solution (over \n'
           +' one steady-state "long" period corresponding to the actual \n'
           +' frequencies) to the average value of the harmonic-balance solution \n'
           +' (over the '+str(N)+' time instances):')
    print('\n\tf_ave (time-accurate) = '+str(round(f_ave_TA,3)))
    print('\tf_ave (harmonic-balance) = '+str(round(f_ave_HB,3)))
    print()

#-----------------------------------------------------------------------------#

# standard boilerplate to call the main() function
if __name__ == '__main__':
    main()
