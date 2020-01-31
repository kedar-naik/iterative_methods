# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 08:37:08 2017

@author: Kedar
"""

import numpy as np
from matplotlib import pyplot as plt
import webbrowser

#-----------------------------------------------------------------------------#
def f(t, a_0, a_1, b_1, omega_1):
    '''
    a single-toned signal
    '''
    f = a_0 + a_1*np.cos(omega_1*t) + b_1*np.sin(omega_1*t)
    return f
#-----------------------------------------------------------------------------#

# define a single-toned signal
omega_1 = 3.0   # [rad/s]

a_0 = 6.0   # mean value
a_1 = 5.0   # amplitude of cosine component
b_1 = 4.0   # amplitude of sine component

T_1 = 2.0*np.pi/omega_1

t_fine = np.linspace(0, 1.5*T_1, 200)
f_fine = f(t_fine, a_0, a_1, b_1, omega_1)

print('\n\tT_1 = ', np.round(T_1,3), 's')

K = 1 
N_0 = 2*K+1
N = N_0
t_T1 = np.arange(N)*T_1/N
f_T1 = f(t_T1, a_0, a_1, b_1, omega_1)

alpha = 1.01
t_Nyquist = np.arange(N)*np.pi/(alpha*omega_1)
f_Nyquist = f(t_Nyquist, a_0, a_1, b_1, omega_1)

# plotting preliminaries
plt.ioff()
plot_name = 'single_toned'
plot_name = 'single_toned_T1'
plot_name = 'single_toned_Nyquist'
auto_open = True
the_fontsize = 16
plt.figure(plot_name)
# plotting
plt.plot(t_fine,f_fine,'k-')
plt.plot([T_1, T_1], [min(f_fine), max(f_fine)], 'r--')
#plt.plot(t_T1, f_T1, 'ko')                  # T1
plt.plot(t_Nyquist, f_Nyquist, 'ko')        # Nyquist
plt.xlabel('$t$', fontsize=the_fontsize)
plt.ylabel('$f\\left(t\\right)$', fontsize=the_fontsize)
# save plot and close
print('\n\t'+'saving final image...', end='')
file_name = plot_name+'.png'
plt.savefig(file_name, dpi=300)
print('figure saved: '+plot_name)
plt.close(plot_name)
# open the saved image, if desired
if auto_open:
    webbrowser.open(file_name)