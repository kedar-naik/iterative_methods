# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 00:35:20 2017

@author: Kedar
"""
import numpy as np
import matplotlib                        # import by itself first
matplotlib.use('Agg')                    # use Anti-Grain Geometry backend
from matplotlib import pylab as plt      # must be called AFTER use()
from matplotlib import animation         # for specifying the writer
plt.close('all')
import webbrowser

# Newton's law of cooling ODE (prob. 18, pg. 63, Boyce & DiPrima)
# Consider an insulated box (a building, perhaps) with internal temperature 
# u(t). According to Newton's law of cooling, u satisfies the differential 
# equation
#                   du
#                   -- = -k*(u(t)-T(t)),
#                   dt
#
# where T(t) varies sinusoidally; for example, assume that 
# T(t) = T_0 + T_1*cos(omega*t)
# Newton's law of cooling reflects early observations made by Newton regarding
# thermal conduction. it is a discrete analogue of the more formal description
# given by Fourier's law.

# constants
t_0 = 0.0               # initial time [hours]
t_end = 3*24.0          # final time [hours]
T_0 = 60.0              # mean ambient temp [degrees F]
T_1 = 15.0              # amplitude of ambient temp variations [degrees F]
k = 0.2                 # constant [1/hr]
omega = 2.0*np.pi/24.0  # corresonds to one ambient-temp cycle per day [rad/hr]
u_0 = 32.0             # set an inital condition inside the box [degrees F]

# discretize the time domain
t_exact = np.linspace(t_0, t_end, 1000)
t_numerical = np.linspace(t_0, t_end, 50)
delta_t = t_numerical[1]-t_numerical[0]

# exact (particular solution with u(0)=u_0)
transient = np.exp(-k*t_exact)*(u_0-T_0-(k**2/(k**2+omega**2))*T_1)
steady_state = T_0 + (k/(k**2+omega**2))*T_1*(k*np.cos(omega*t_exact)+omega*np.sin(omega*t_exact))
u_exact = transient + steady_state

# try multiple initial conditions
n_curves = 10
u_0s = np.linspace(-40.0,80,n_curves)
u_0s = np.concatenate((u_0s,[u_0]))
u_curves_exact = []
u_curves_numerical = []
for u_0_k in u_0s:
    # numerical solution
    u_numerical_k = [u_0_k]
    for i in range(len(t_numerical)-1):
        T_previous = T_0 + T_1*np.cos(omega*t_numerical[i])
        u_next = u_numerical_k[i] + delta_t*(-k*(u_numerical_k[i]-T_previous))
        u_numerical_k.append(u_next)
    u_curves_numerical.append(u_numerical_k)
    # exact solution
    transient_k = np.exp(-k*t_exact)*(u_0_k-T_0-(k**2/(k**2+omega**2))*T_1)
    steady_state_k = T_0 + (k/(k**2+omega**2))*T_1*(k*np.cos(omega*t_exact)+omega*np.sin(omega*t_exact))
    u_exact_k = transient_k + steady_state_k
    u_curves_exact.append(u_exact_k)

# plotting: USER INPUTS! do you want to animate the solution history or just
# plot the final result? (True = animate, False = just print final result)
animate_plot = False
plot_name = 'cooling_law'
n_images = len(t_exact)                  # total number of images computed
skip_images = 10                   # images to skip between animation frames

# plotting: initializations
fig = plt.figure()
# plotting: things that will not be changing inside the loop

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
        # clear the previous line and make the new plot
        plt.cla()
        plt.plot(t_exact[:n],u_exact[:n],'k-')
        plt.xlabel('$t, \\left[hr\\right]$',fontsize=14)
        plt.ylabel('$u(t), \\left[^\\degree F\,\\right]$', fontsize=14)
        plt.xlim(0,80)
        plt.ylim(-40,80)
        # progress monitor
        percent_done = float(n)*100.0/(n_images-1)
        print('capturing fig. '+plot_name+' (frame #'+str(frame)+'): ', \
               round(percent_done,2),'%')
        writer.grab_frame()
        frame += 1
    writer.grab_frame()
# plotting: save an image of the final frame
print('\n'+'saving final image...')
plt.savefig(plot_name, dpi=400)
print('figure saved: ' + plot_name)
# free memory used for the plot
plt.close(fig)

# print some information to the screen
print('\n\tdelta_t =', delta_t)

# plot the solution
plt.figure()
#plt.plot(t_exact,u_exact,'k-', label='$full \; solution$')
#plt.plot(t_exact,transient, 'r--', label='$transient$')
#plt.plot(t_exact,steady_state, 'b-.', label = '$steady \; state$')
plt.plot(t_numerical,u_curves_numerical[-1],'g.-',label='$numerical \; sol.$')  
plt.xlabel('$t, \\left[hr\\right]$',fontsize=14)
plt.ylabel('$u(t), \\left[^\\degree F\,\\right]$', fontsize=14)
plt.ylim([-40,80])
#plt.legend(loc='best')
plt.savefig('u_vs_t.png', dpi=400)
plt.close()
webbrowser.open('u_vs_t.png')

# plot the solution
plt.figure()
for k in range(n_curves):
    if k==0:
        plt.plot(t_exact,u_curves_exact[k],'m',label='$exact \; solutions.$')
        #plt.plot(t_numerical,u_curves_numerical[k],'g',label='$numerical \; solutions.$')
    else:
        plt.plot(t_exact,u_curves_exact[k],'m')
        #plt.plot(t_numerical,u_curves_numerical[k],'g')
plt.xlabel('$t, \\left[hr\\right]$',fontsize=14)
plt.ylabel('$u(t), \\left[^\\degree F\,\\right]$', fontsize=14)
#plt.legend(loc='best') 
plt.savefig('u_vs_t_curves.png', dpi=400)
plt.close()
webbrowser.open('u_vs_t_curves.png')