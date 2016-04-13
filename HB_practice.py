# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 01:16:35 2016

@author: Kedar
"""

import math
import pylab as plt
from time_spectral import myLinspace
from matplotlib import animation         # for specifying the writer


freqs = [1.0, 2.3, 3.8]

#-----------------------------------------------------------------------------#
def my_non_periodic_fun(t,freqs):
    '''
    given a set of specfied frequencies and time points, this subroutine 
    samples a function composed of a sum of sinusoids oscillating at the 
    frequencies specified
    '''
    f = []
    # run through all the time samples
    for t_i in t:
        f_i = 0
        # run through all the frequencies
        for freq in freqs:
            f_i = f_i + math.sin(freq*t_i)
        f.append(f_i)
    return f
#-----------------------------------------------------------------------------#
def my_non_periodic_ode(t, u, freqs):
    
    '''
    given a set of specfied frequencies and time points, this subroutine 
    samples a function (meant the represent the RHS of an ODE) composed of a 
    sum of sinusoids oscillating at the frequencies specified
    '''
    import math
    
    # The equation for periodic population harvesting
    k = 0.5                   # growth rate of population (Malthusian param.)
    C = 10                    # carrying capacity
    # expression giving the derivative    
    dudt = k*u*(1-(u/C))
    # run through all the frequencies
    for freq in freqs:
        dudt = dudt - math.sin(freq*t)
    return dudt
#-----------------------------------------------------------------------------#
    
# time period corresponding to the lowest frequency
T_lowest_freq = (2.0*math.pi)/min(freqs)

# discretize that longest time period
t = myLinspace(0, T_lowest_freq,100*math.ceil(T_lowest_freq))
f = my_non_periodic_fun(t,freqs)
    
# find the time instances here
K = len(freqs)
N = 2*K+1
delta_t = T_lowest_freq/N
t_HB = [i*delta_t for i in range(N)]
f_HB = my_non_periodic_fun(t_HB,freqs)

# plot everything
plt.close('all')
plt.figure()
plt.plot(t,f)
plt.plot(t_HB,f_HB,'ko')
plt.xlabel('$t$', fontsize=16)
plt.ylabel('$f(t)$', fontsize=16)


###########################################################################
# [time accurate] explicit euler ##########################################
###########################################################################
delta_t = 0.05             # time step
initial_value = 2.0        # intitial condition

t_start = 0                # initial time
t_end = 90                 # approx. final time (stop at or just after)
  
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
        # explicitly step forward in time 
        f.append(f[n-1] + delta_t*my_non_periodic_ode(times[n-1],f[n-1],freqs))
        
# plotting: USER INPUTS! do you want to animate the solution history or just
# plot the final result? (True = animate, False = just print final result)
animate_plot = True                   
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