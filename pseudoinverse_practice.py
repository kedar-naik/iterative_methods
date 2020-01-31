# -*- coding: utf-8 -*-

"""

use of the pseudoinverse

Created on Sun Jul 31 23:36:52 2016

@author: Kedar
"""
import numpy as np
from matplotlib import pyplot as plt
import webbrowser
from iterative_methods import my_inv, my_pinv

# turn off interactive mode, so that plot windows don't pop up
#plt.ioff()
# close all open figures
plt.close('all')

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
        f_i = 0 - 15
        df_i = 0
        # run through all the frequencies
        for omega in omegas:
            f_i += np.cos(omega*t_i) + 3.0*np.sin(omega*t_i)**2
            df_i += omega*np.cos(omega*t_i)
        f.append(f_i)
        df.append(df_i)
    # turn f and df into numpy column vectors
    f = np.array(f)
    f = f.reshape(f.size,1)
    df = np.array(df)
    df = df.reshape(df.size,1)
    return f, df
#-----------------------------------------------------------------------------#

# given angular frequencies
omegas = [1.5, 2.3]

# set the time discretization
time_discretization = 'use_T1'

# create the harmonic balance operator matrix and find the time instances
D_HB, t_HB = harmonic_balance_operator(omegas, time_discretization)

# invert D_HB
D_HB_inv = np.linalg.inv(D_HB)
D_HB_inv = my_inv(D_HB)

# compute the pseudoinverse
D_HB_pinv = np.linalg.pinv(D_HB)
D_HB_pinv = my_pinv(D_HB)

# compute the function values
f_HB, dummy = my_non_periodic_fun(t_HB, omegas)

# find the average function value
f_HB_mean = np.average(f_HB)

# derivative of the function
df_HB = np.dot(D_HB,f_HB)

# invert D_HB and try to get back the function (this shouldn't work)
# (this will not give the right answer, as D_HB in singular)
f_HB_inv = np.dot(D_HB_inv,df_HB)

# try solving for f_HB using the pseudoinverse
# (this is the shortest-length least-squares solution)
f_HB_pinv = np.dot(D_HB_pinv,df_HB)

# add the function average to f_HB_pinv
f_HB_pinv_plus_mean = f_HB_pinv + f_HB_mean

# use an average of the two
f_HB_inv_pinv_ave = 0.5*np.dot(D_HB_inv+D_HB_pinv,df_HB)


# plot the function
plot_name = 'pseudoinverse_analysis'
auto_open = True
plt.figure(plot_name)
plt.plot(t_HB,f_HB,'ko-',label='$f_{HB}$')
plt.plot(t_HB,df_HB,'go-',label='$\\frac{\partial f_{HB}}{\partial t}=D_{HB}f_{HB}$')
plt.plot(t_HB,f_HB_inv,'ro--',label='$D_{HB}^{-1}\\frac{\partial f_{HB}}{\partial t}$')
plt.plot(t_HB,f_HB_pinv,'bo--',label='$D_{HB}^+\\frac{\partial f_{HB}}{\partial t}$')
plt.plot(t_HB,f_HB_pinv_plus_mean,'co--',label='$D_{HB}^+\\frac{\partial f_{HB}}{\partial t} + \\overline{f_{HB}}$')
plt.plot(t_HB,f_HB_inv_pinv_ave,'mo--',label='$\\frac{1}{2}\\left( D_{HB}^{-1} + D_{HB}^+ \\right)\\frac{\partial f_{HB}}{\partial t}$')
plt.xlabel('$t$', fontsize=16)
plt.ylabel('$f$', fontsize=16)
plt.title('$D_{HB}f_{HB}=\\frac{\partial f_{HB}}{\partial t}$', y=1.03)
plt.legend(loc='best',fontsize=6)

# save plot and close
print('\n\t'+'saving final image...', end='')
file_name = plot_name+'.png'
plt.savefig(file_name, dpi=300)
print('figure saved: '+plot_name)
plt.close(plot_name)
# open the saved image, if desired
if auto_open:
    webbrowser.open(file_name)

# print rank of the operator matrix
print('\nD_HB =\n '+str(np.around(D_HB,2))[1:-1])
print('\nD_HB_inv =\n '+str(np.around(D_HB_inv,2))[1:-1])
print('\nD_HB_pinv =\n '+str(np.around(D_HB_pinv,2))[1:-1])

print('\nrank(D_HB) = '+str(np.linalg.matrix_rank(D_HB)))
print('rank(D_HB_inv) = '+str(np.linalg.matrix_rank(D_HB_inv)))
print('rank(D_HB_pinv) = '+str(np.linalg.matrix_rank(D_HB_pinv)))