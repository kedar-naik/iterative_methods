# -*- coding: utf-8 -*-
"""
showing that harmonic intervals lead to periodic signals

Created on Tue Sep 19 21:08:48 2017

@author: Kedar
"""

import numpy as np
from matplotlib import pyplot as plt
import webbrowser
from fractions import Fraction
from HB_practice import period_given_freqs

# frequencies
omega_1 = 5.0   # rad/s
omega_2 = (3/2)*omega_1   # rad/s (perfect fifth)
omega_2 = (45/32)*omega_1   # rad/s (tritone)

# harmonic interval
omega_ratio = Fraction(omega_2/omega_1)    # returns a Fraction object (a,b)

# overall period
T_long = period_given_freqs([omega_1, omega_2])

# signal
t = np.linspace(-1.5*T_long,1.5*T_long,20*4000)
f = np.sin(5.0*t) + np.sin(7.5*t)

# test periodicity
f_at_zero = np.interp(0.0,t,f)
f_at_T_long = np.interp(T_long,t,f)
f_difference = f_at_T_long-f_at_zero
eps = np.finfo(float).eps

# print everything
print('\n\tomega_1 = '+str(omega_1)+' rad/s')
print('\n\tomega_2 = '+str(omega_2)+' rad/s')
print('\n\tomega ratio = '+str(omega_ratio))   # str(Fraction)='a/b'
print('\n\tT = '+str(T_long)+' s')
print('\n\tf(T)-f(0) = '+str(f_difference))
print('\n\tmachine epsilon = '+str(eps)+'\n')

# plot everything
plot_name = 'harmonic_periodicity'
auto_open = True
plt.figure(plot_name)
plt.plot(t, f, label='$f$')
plt.xlim(-1.5*T_long, 1.5*T_long)
plt.xlim(-3.5, 3.5)
plt.xlabel('$t, \\left[\,\\mathrm{s}\,\\right]$',fontsize=14)
plt.ylabel("$f(t)$", fontsize=14)
#plt.ylabel("$f'(t)$", fontsize=14)     # uncomment if running tritone!
plt.grid()
#plt.legend(loc='best')
#plt.title('$\\frac{\omega_2}{\omega_1} = \\frac{'+str(omega_ratio.numerator)+'}{'+str(omega_ratio.denominator)+'} \qquad T='+str(T_long)+'$', fontsize=12, y=1.03)
# save plot and close
print('\n\t'+'saving final image...', end='')
file_name = plot_name+'.png'
plt.savefig(file_name, dpi=300)
print('figure saved: '+plot_name)
plt.close(plot_name)
# open the saved image, if desired
if auto_open:
    webbrowser.open(file_name)