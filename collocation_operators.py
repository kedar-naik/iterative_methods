# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 00:38:49 2017

@author: Kedar
"""

import numpy as np
from matplotlib import pyplot as plt
import webbrowser

#-----------------------------------------------------------------------------#
def collocation_operator(N, order=1):
    '''
    this subroutine returns the Fourier collocation spectral differentiation
    matrix for either first or second derivatives. the N collocation points are
    assumed to have been sampled over the domain [0, 2*pi]. Expressions taken
    from Peyret 2002, ch. 2.
    '''
    # initialize the differentiation matrix
    D = np.empty((N,N))
    # assuming N uniformly spaced points just shy of spanning [0,2*pi]
    delta = 2.0*np.pi/N
    # first derivatives
    if order==1:
        # odd N
        if not N%2==0:
            for i in range(N):
                for j in range(N):
                    if i==j:
                        D[i,j] = 0.0
                    else:
                        # set the sign for h(i,j)
                        if i > j:
                            h_ij = delta/2
                        else:
                            h_ij = -delta/2
                        D[i,j] = (-1)**(i+j)/(2.0*np.sin(h_ij))
        # even N
        else:
            for i in range(N):
                for j in range(N):
                    if i==j:
                        D[i,j] = 0.0
                    else:
                        # set the sign for h(i,j)
                        if i > j:
                            h_ij = delta/2
                        else:
                            h_ij = -delta/2
                        D[i,j] = 0.5*(-1)**(i+j)/np.tan(h_ij)
    # second derivatives
    if order==2:
        # odd N
        if not N%2==0:
            for i in range(N):
                for j in range(N):
                    if i==j:
                        D[i,j] = -(N**2-1)/12
                    else:
                        # set the sign for h(i,j)
                        if i > j:
                            h_ij = delta/2
                        else:
                            h_ij = -delta/2
                        D[i,j] = (-1)**(i+j+1)*np.cos(h_ij)/(2.0*np.sin(h_ij)**2)
        # even N
        else:
            for i in range(N):
                for j in range(N):
                    if i==j:
                        D[i,j] = -(N-1)*(N-2)/12
                    else:
                        # set the sign for h(i,j)
                        if i > j:
                            h_ij = delta/2
                        else:
                            h_ij = -delta/2
                        D[i,j] = 0.25*(-1)**(i+j)*N + (-1)**(i+j+1)/(2.0*np.sin(h_ij)**2)
    return D
#-----------------------------------------------------------------------------#

# select the number of collocation points
N = 9

# find the locations of these N points, uniformly spacing over [0, 2*pi]
t_tilde = np.array([j*2*np.pi/N for j in range(N)])

# pick a period for your periodic function
T = 2*np.pi
# compute the location of the sample points over this period
t = T/(2*np.pi)*t_tilde
# sample a periodic function at the time instances
f = np.sin(2*np.pi*t/T)
# convert this into a column vector
f = np.reshape(f,(N,1))

# compute a fine-grid version of the function and its derivative
t_fine = np.linspace(0,T)
f_fine = np.sin(2*np.pi*t_fine/T)
df_fine = (2*np.pi/T)*np.cos(2*np.pi*t_fine/T)
df2_fine = -(2*np.pi/T)**2*np.sin(2*np.pi*t_fine/T)

# build the differentiation matrix (for first derivatives)
D = collocation_operator(N)

# apply the operator
df = np.dot(D,f)

# build the differentiation matrix (for second derivatives)
D = collocation_operator(N, order=2)

# apply the operator
df2 = np.dot(D,f)

# plot the time history
plot_name = 'collocation_check'
auto_open = True
fig = plt.figure(plot_name)
width, height = fig.get_size_inches()
fig.set_size_inches(3.5*width, height, forward=True)
plt.suptitle('$N='+str(N)+'$\n', fontsize=16)
plt.subplot(1,3,1)
plt.plot(t_fine, f_fine, 'r-', label='$f(t)$')
plt.plot(t, f, 'bo', label='$f_{TS}$')
plt.xlabel('$t$', fontsize=18)
plt.ylabel('$f(t)$', fontsize=18)
plt.legend(loc='best')
plt.subplot(1,3,2)
plt.plot(t_fine, df_fine, 'r-', label='$\\dot{f}(t)$')
plt.plot(t, df, 'bo', label='$\\dot{f}_{TS}$')
plt.xlabel('$t$', fontsize=18)
plt.ylabel('$\\dot{f}(t)$', fontsize=18)
plt.legend(loc='best')
plt.subplot(1,3,3)
plt.plot(t_fine, df2_fine, 'r-', label='$\\ddot{f}(t)$')
plt.plot(t, df2, 'bo', label='$\\ddot{f}_{TS}$')
plt.xlabel('$t$', fontsize=18)
plt.ylabel('$\\ddot{f}(t)$', fontsize=18)
plt.legend(loc='best')
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
    





