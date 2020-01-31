# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 00:11:31 2017

@author: Kedar
"""

import numpy as np
import webbrowser
from matplotlib import pyplot as plt
from HB_practice import my_non_periodic_fun



actual_omegas = [1.0, 2.1, 3.6]         # all freqs admissible
omegas = list(actual_omegas)

# test with some noise
actual_omegas = [1.02, 2.31, 3.86]         # all freqs admissible
omegas = [1.0, 2.1, 3.6]

# set W (the bandwidth) equal to omega_K
W = 1.001*omegas[-1]      # [rad/s]

# there are 2L+1 samples to use
L = 655

# set the time points where the samples need to be taken
T1 = 2.0*np.pi/omegas[0]
delta_t = T1/(2.0*L+1.0)
t_s = [j*delta_t for j in range(1,2*L+1+1)]

# create the operator matrix
D_s = np.zeros((2*L+1,2*L+1))
for j in range(2*L+1):
    for k in range(-L,L+1):
        chi_jk = 2.0*W*t_s[j]-k
        D_s[j][k] = 2.0*W*(chi_jk*np.cos(chi_jk)-np.sin(chi_jk))/(chi_jk**2)

# sample the signal at the time-sample points
f_at_s, df_at_s = my_non_periodic_fun(t_s, actual_omegas, verbose=False)

# create a fine time grid
N_fine = 200
t_fine = np.linspace(-1.25*L*delta_t, T1, N_fine)
# sample the exact signal on the fine grid
f_fine, df_fine = my_non_periodic_fun(t_fine, actual_omegas, verbose=False)

# apply the sinc operator matrix to compute the time derivative 
df_s = np.dot(D_s, f_at_s)

# create a longer operator matrix to produce a fine-grid projection of df_s
D_fine = np.zeros((N_fine,2*L+1))
for j in range(N_fine):
    for k in range(-L,L+1):
        chi_jk = 2.0*W*t_fine[j]-k
        D_fine[j][k] = np.sin(chi_jk)/chi_jk

# apply the projection operator to the solved points
df_s_fine = np.dot(D_fine, df_s)


# plot everything
plot_name = 'sinc_operator_check - L='+str(L)
auto_open = True
plt.figure(plot_name)
plt.plot(t_fine, f_fine, label='$f_{exact}$')
plt.plot(t_fine, df_fine, 'r-', label='$df/dt_{exact}$')
plt.plot(t_s, f_at_s, 'k.', label='$f_{sinc}$')
plt.plot(t_s, df_s, 'g.', label='$df/dt_{sinc}$')
plt.plot(t_s, df_s, 'g--')
#plt.plot(t_fine, df_s_fine, 'g--', label='$sinc\,\,interp.$')
plt.xlabel('$t$', fontsize=14)
plt.ylabel('$f(t)$', fontsize=14)
plt.legend(loc='best')
plt.title('$\omega_{actual} = \{' + str(actual_omegas)[1:-1] + \
          '\} \quad\quad \omega_{1} = '+str(omegas[0]) + \
          '\quad\quad \omega_{K} = '+str(omegas[-1]) + \
          '\quad\quad L = '+str(L) + '$', fontsize=12)
# save plot and close
print('\n\t'+'saving final image...', end='')
file_name = plot_name+'.png'
plt.savefig(file_name, dpi=300)
print('figure saved: '+plot_name)
plt.close(plot_name)
# open the saved image, if desired
if auto_open:
    webbrowser.open(file_name)
