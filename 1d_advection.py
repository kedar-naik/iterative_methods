# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 22:30:54 2016

@author: Kedar
"""
import math
import matplotlib                        # import by itself first
matplotlib.use('Agg')                    # use Anti-Grain Geometry backend
from matplotlib import pylab as plt      # must be called AFTER use()
from matplotlib import animation         # for specifying the writer

# setting up the 1D advection problem
x_start = 0.0       # [m]
x_end = 10.0        # [m]
v = 5.0             # [m/s]

# number of points discretizing the x domain
N = 200

# number of time steps to take
N_timesteps = 1000

# desired CFL (must be less than 1 for stabilty)
desired_CFL = 0.85

# spatial grid
delta_x = (x_end-x_start)/(N-1)
x = [x_start + i*delta_x for i in range(N)]

# Dirichelet boundary conditions at ends
u_start = 0.0
u_end = 0.0

# define Gaussian bump
mean = 1.5
variance = 0.1
a = 1/(math.sqrt(variance)*math.sqrt(2.0*math.pi))
b = mean
c = math.sqrt(variance)

# initial solution
u = [a*math.exp(-pow(x_i-b,2.0)/(2.0*pow(c,2.0))) for x_i in x]

# enforce Dirichelet boundary conditions
u[0] = u_start      # u(x_start) = 0
u[-1] = u_end       # u(x_end) = 0

# append initial solution to history
u_hist = []
u_hist.append(u)
t_hist = [0]

# max time step required for stabilty
delta_t = desired_CFL*delta_x/v

# compute CFL number
CFL = v*delta_t/delta_x

# march teh solution forward using the Lax method
for n in range(N_timesteps):
    u_next = [0]*len(x)
    u = u_hist[n]
    for i in range(N):
        if i == 0:
            #u_next[i] = u_start
            u_next[i] = -0.5*math.cos(2.0*math.pi*t_hist[-1]/0.25) + 0.5
        elif i == N-1:
            u_next[i] = u_end
        else:
            u_next[i] = 0.5*(u[i+1]+u[i-1]) - 0.5*CFL*(u[i+1]-u[i-1])
    u_hist.append(u_next)
    t_hist.append(t_hist[-1]+delta_t)
    # progress monitor
    percent_done = float(n)*100.0/N_timesteps
    print('computing solution... '+str(percent_done)+'%')

# plotting: USER INPUTS! do you want to animate the solution history or just
# plot the final result? (True = animate, False = just print final result)
animate_plot = True                 
plot_name = '1d_advection'
n_images = N_timesteps+1            # total number of images computed
skip_images = 10                   # images to skip between animation frames

# plotting: initializations
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.close('all')
fig = plt.figure()
l, = plt.plot([], [],'k.-',label='u')
# plotting: things that will not be changing inside the loop
u_min = 1e10
u_max = -1e10
for u_sol in u_hist:
    for u_i in u_sol:
        if u_i > u_max:
            u_max = u_i
        if u_i < u_min:
            u_min = u_i
vertical_padding = (u_max-u_min)/4.0
plt.ylim(u_min-vertical_padding,u_max+vertical_padding)
# plotting: set the total number of frames
if animate_plot == True:
    # capture all frames (skipping, if necessary) and the final frame
    all_frames = list(range(0,n_images,skip_images+1))+[n_images-1]
else:
    # no animation: just capture the last one
    all_frames = [n_images-1]
# plotting: capturing the movie
writer = animation.writers['ffmpeg'](fps=15)
with writer.saving(fig, plot_name+'.mp4', 100):
    frame = 0
    for n in all_frames:
        plt.clf()
        plt.plot(x,u_hist[n],'k.-')  
        plt.xlabel('$x$', fontsize=18)
        plt.ylabel('$u(x,t)$', fontsize=18)
        plt.xlim(x_start,x_end)
        plt.ylim(u_min-vertical_padding,u_max+vertical_padding)
        plt.title('$'+r'\frac{\partial u}{\partial t} = -v \frac{\partial u}{\partial x} \quad CFL = '+str(round(CFL,2))+'\quad t = '+str(round(t_hist[n],2))+'s$')
        # Make room for the ridiculously large title.
        plt.subplots_adjust(top=0.8)
        # progress monitor
        percent_done = float(n)*100.0/(n_images-1)
        print('capturing fig. '+plot_name+' (frame #'+str(frame)+'): ', \
               percent_done,'%')
        writer.grab_frame()
        frame += 1
    writer.grab_frame()
# plotting: save an image of the final frame
print('\n'+'saving final image...')
plt.savefig(plot_name, dpi=400)
print('figure saved: ' + plot_name)
# free memory used for the plot
plt.close(fig)

# print output to the screen
print('\n\tdelta_x = ', delta_x)
print('\tdelta_t = ', delta_t)
print('\tv = ', v)
print('\tCFL = ', CFL)


 
