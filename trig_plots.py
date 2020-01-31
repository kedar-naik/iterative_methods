# -*- coding: utf-8 -*-
"""
Created on Mon May 15 15:20:16 2017

@author: 1077177
"""

from matplotlib import pyplot as plt
import webbrowser
import numpy as np

# useage: for the dashsed-line plots, set inadmissible_omegas = []

# CASE I.A: assumed set of angular frequencies [rad/s]
omegas = [1.0, 2.1, 3.5]

# list of inadmissible frequencies under T1 sampling
inadmissible_omegas = [3.5]

# CASE I.A: assumed set of angular frequencies [rad/s]
omegas = [1.0, 2.1, 10.5]

# list of inadmissible frequencies under T1 sampling
inadmissible_omegas = [10.5]

# CASE I.B
#omegas = [1.0, 2.1, 7.0]
#inadmissible_omegas = [7.0]

#omegas = [1.0, 2.1, 14.0-1.75]
#inadmissible_omegas = [14.0-1.75]

# CASE II
#omegas = [1.0, 2.1, 4.9]
#inadmissible_omegas = [2.1, 4.9]

# admissible set
#omegas = [1.0, 2.1, 3.6]
inadmissible_omegas = []


# the number of angular frequencies provided
K = len(omegas)

# the number of time instances used
N = K+1
N = 2*K+1
#N = 3*K+1

# compute the Nyquist frequency (folding frequency) arising from uniformly
# sampling the range [0,T1) using N points (from derivation)
folding_omega = 0.5*min(omegas)*N

# if any of the assumed omegas (beyond the first one) lie at or beyond the folding
# frequency, then compute the aliased frequency
aliased_omegas = [min(omegas)]
for omega in omegas[1:]:
    if omega >= folding_omega:
        aliased_omegas.append(omega-2.0*folding_omega)
    else:
        aliased_omegas.append(omega)

# the N time instances, evenly spaced over T1
T1 = 2.0*np.pi/min(omegas)
delta_t = T1/N
t_HB = np.array([j*delta_t for j in range(N)])

# fine time grid and colors for plotting
t = np.linspace(0.0, 1.0*T1, 500)
colors = ['r','g','b','m']

plot_abs_value_sin = False

the_fontsize = 16

# plotting
plt.ioff()
plot_name = 'prescribed frequencies'
auto_open = True
plt.figure(plot_name)
# run through the omegas
counter = 0
for omega in omegas:
    # pick a color for this omega
    color = colors[counter]
    # set the different markers
    if omega in inadmissible_omegas:
        fine_marker = '--'
        discrete_marker = 'o-'
    else:
        fine_marker = '-'
        discrete_marker = 'o'
    # plot the cosine component
    plt.subplot(2,1,1)
    plt.plot(t, np.cos(omega*t),color+fine_marker, label='$\\omega_'+str(counter+1)+'='+str(omegas[counter])+'\;\\frac{rad}{s}$')
    plt.plot(t_HB, np.cos(omega*t_HB), color+discrete_marker)
    # plot the sine component
    plt.subplot(2,1,2)
    if plot_abs_value_sin:
        plt.plot(t, np.abs(np.sin(omega*t)),color+fine_marker)
        plt.plot(t_HB, np.abs(np.sin(omega*t_HB)), color+discrete_marker)
        plt.ylabel('$|sin(\\omega_k t)|$', fontsize=the_fontsize)
        plt.title('$|\\mathrm{Im}\{e^{i\\omega_k t}\}|$', fontsize=the_fontsize)
    else:
        plt.plot(t, np.sin(omega*t),color+fine_marker)
        plt.plot(t_HB, np.sin(omega*t_HB), color+discrete_marker)
        plt.ylabel('$sin(\\omega_k t)$', fontsize=14)
        plt.title('$\\mathrm{Im}\{e^{i\\omega_k t}\}$', fontsize=the_fontsize)
    # increment the counter
    counter += 1
plt.subplot(2,1,1)
plt.xlim(0.0,T1)
plt.ylim(-1.1, 1.1)
plt.xlabel('$t, [s]$', fontsize=the_fontsize)
plt.ylabel('$cos(\\omega_k t)$', fontsize=the_fontsize)
plt.legend(loc='right')
plt.title('$\\mathrm{Re}\{e^{i\\omega_k t}\}$', fontsize=the_fontsize)
plt.subplot(2,1,2)
plt.xlim(0.0,T1)
plt.ylim(-1.1, 1.1)
plt.xlabel('$t, [s]$', fontsize=the_fontsize)
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

if not omegas == aliased_omegas:
    # plotting
    plot_name = 'aliased frequencies'
    auto_open = True
    plt.figure(plot_name)
    # run through the omegas
    counter = 0
    for omega in aliased_omegas:
        # pick a color for this omega
        color = colors[counter]
        # set the different markers
        if omega in inadmissible_omegas:
            fine_marker = '--'
            discrete_marker = 'o-'
        else:
            fine_marker = '-'
            discrete_marker = 'o'
        # plot the cosine component
        plt.subplot(2,1,1)
        plt.plot(t, np.cos(omega*t),color+fine_marker, label='$\\omega_'+str(counter+1)+'='+str(round(aliased_omegas[counter],2))+'\;\\frac{rad}{s}$')
        plt.plot(t_HB, np.cos(omega*t_HB), color+discrete_marker)
        # plot the sine component
        plt.subplot(2,1,2)
        if plot_abs_value_sin:
            plt.plot(t, np.abs(np.sin(omega*t)),color+fine_marker)
            plt.plot(t_HB, np.abs(np.sin(omega*t_HB)), color+discrete_marker)
            plt.ylabel('$|sin(\\omega_k t)|$', fontsize=the_fontsize)
            plt.title('$| \\mathrm{Im}\{e^{i\\omega_k t}\}|$', fontsize=the_fontsize)
        else:
            plt.plot(t, np.sin(omega*t),color+fine_marker)
            plt.plot(t_HB, np.sin(omega*t_HB), color+discrete_marker)
            plt.ylabel('$sin(\\omega_k t)$', fontsize=the_fontsize)
            plt.title('$\\mathrm{Im}\{e^{i\\omega_k t}\}$', fontsize=the_fontsize)
        # increment the counter
        counter += 1
    plt.subplot(2,1,1)
    plt.xlim(0.0,T1)
    plt.ylim(-1.1, 1.1)
    plt.xlabel('$t, [s]$', fontsize=the_fontsize)
    plt.ylabel('$cos(\\omega_k t)$', fontsize=the_fontsize)
    plt.legend(loc='right')
    plt.title('$\\mathrm{Re}\{e^{i\\omega_k t}\}$', fontsize=the_fontsize)
    plt.subplot(2,1,2)
    plt.xlim(0.0,T1)
    plt.ylim(-1.1, 1.1)
    plt.xlabel('$t, [s]$', fontsize=the_fontsize)
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