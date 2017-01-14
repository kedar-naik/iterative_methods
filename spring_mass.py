# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 15:09:25 2016

@author: 1077177
"""

from matplotlib import pyplot as plt
import math
import webbrowser               # for opening files automatically

# turn off interactive mode, so that plot windows don't pop up
plt.ioff()

# spring definition
x_eq = 2                    # equilibrium spring length, [m]
k = 4900                    # spring constant, [N/m]

# wagon definition
m = 250                     # wagon mass, [kg]

# intial conditions
x_0 = x_eq                  # intial wagon position, [N/m]
v_0 = 1                     # intial wagon velocity, [m/s]
t_0 = 0                     # intial time, [s]

# time-step definition
del_t = 0.0001              # time step, [s]
t_end = 10                  # final time, [s]

# create a dictionary defining the problem
my_spring_problem = {'m': m,           # mass, [kg]
                     'k': k,           # spring constant, [N/m] 
                     'x_eq': x_eq,     # equilibrium spring length, [m]
                     'x_0': x_0,       # intial mass position, [N/m]
                     'v_0': v_0,       # intial mass velocity, [m/s]
                     't_0': t_0}       # intial time, [s]


#-----------------------------------------------------------------------------#

def myCart2pol(x, y):
    '''
    This function converts Cartesian coordinates to polar coordinates.
    '''
    import math
    
    rho = math.sqrt(x**2 + y**2)
    phi = math.atan2(y, x)
    
    return(rho, phi)

#-----------------------------------------------------------------------------#

def myPol2cart(rho, phi):
    '''
    This function converts polar coordinates to Cartesian coordinates.
    '''
    
    import math
    
    x = rho*math.cos(phi)
    y = rho*math.sin(phi)
    
    return(x, y)

#-----------------------------------------------------------------------------#

def plot_wagon(wagon_plot_name, x, wagon_width):
    '''
    plots a wagon with its back side at the given x location
    '''
    import pylab as plt
    
    # set wagon and wheel dimensions
    wagon_height = wagon_width/2.5    # [m]
    wheel_radius = wagon_height/2.5  # [m]
    
    # draw both wheels
    wheel_angles = [math.pi]
    wheel_points = 100
    for i in range(1, wheel_points):
        wheel_angles.append(wheel_angles[i-1]+math.pi/(wheel_points-1))
    wheel_coords = [myPol2cart(wheel_radius,angle) for angle in wheel_angles]
    x_back_wheel = [coords[0]+x+0.25*wagon_width for coords in wheel_coords]
    x_front_wheel = [coords[0]+x+0.75*wagon_width for coords in wheel_coords]
    y_wheel = [coords[1]+wheel_radius for coords in wheel_coords]
    
    # draw the bottom and top lines of the wagon
    x_horizontal = [x, x+wagon_width]
    y_bottom = [wheel_radius]*2
    y_top = [wheel_radius+wagon_height]*2
    
    # draw the sides of the wagon
    y_vertical = [wheel_radius, wheel_radius+wagon_height]
    x_left = [x]*2
    x_right = [x+wagon_width]*2
    
    # plot the different parts
    plt.figure(wagon_plot_name)
    plt.plot(x_back_wheel, y_wheel, 'k', x_front_wheel, y_wheel, 'k')
    plt.plot(x_horizontal, y_bottom, 'k')
    plt.plot(x_horizontal, y_top, 'k')
    plt.plot(x_left, y_vertical, 'k')
    plt.plot(x_right, y_vertical, 'k')
    plt.fill_between(x_back_wheel,y_wheel,wheel_radius,facecolor='black')
    plt.fill_between(x_front_wheel,y_wheel,wheel_radius,facecolor='black')
    plt.fill_between(x_horizontal,y_bottom,y_top,facecolor='brown')
    plt.axis('equal')
    
    return 0

#-----------------------------------------------------------------------------#
    
def plot_spring(wagon_plot_name, x, flat_end_length, plot_func=False):
    '''
    This function plots a spring stretching from 0 to the given x.
    '''
    import pylab as plt
    import math
    
    # check to make sure flat-end length is allowed
    if 2*flat_end_length >= x:
        raise ValueError('\n The flat-end length is too long! \n')
    
    # spring centerline height
    centerline_height = (flat_end_length*3.0/2.5)*(1/2+1/2.5)
    
    # spring characteristics
    n_coils = 10                            # no. of coils to be plotted
    coil_diameter = flat_end_length         # [m]
    
    # draw the starting and ending flat bits
    x_start = [0, flat_end_length]
    x_end = [x-flat_end_length, x]
    y_flat = [centerline_height]*2
    
    # distance to be spanned by coils
    coiled_span = x-2*flat_end_length
    
    # number of revolutions required to draw the desired number of coils
    coils_to_span = n_coils+0.5
    
    # value of the lifted cosine function at the number of coils required
    # N.B. this function has a period of one. the arguement t is in units of
    # periods. func = 2*t - cos(2*pi*t) + 1
    func_at_span = 2*coils_to_span - math.cos(2*math.pi*coils_to_span) + 1
    
    # compute the scaling factor
    scaling_fac = coiled_span/func_at_span
    
    # draw the coils
    coil_points = 300
    delta_coil = coils_to_span/(coil_points-1)
    t_coils = [0]
    x_coils = [flat_end_length]
    y_coils = [centerline_height]
    for i in range(coil_points):
        # time points
        t_coil = i*delta_coil
        t_coils.append(t_coil)
        # the values computed from the func are stretched to match the coiled-
        # span length and then added to the flat-end length
        x_coils.append(flat_end_length + scaling_fac*(2.0*t_coil \
                       - math.cos(2*math.pi*t_coil) + 1))
        # a sin wave with the amplitude of the coiling radius give the height
        y_coils.append(centerline_height \
                       + coil_diameter/2*math.sin(2*math.pi*t_coil))
    
    # if desired, plot of the function used to plot the x locations
    if plot_func:
        check_plot_name = 'function_check'
        check_filename = check_plot_name+'.png'
        plt.figure()
        plt.plot(t_coils,x_coils)
        plt.xlabel('t*')
        plt.ylabel('x*')
        # save plot and close
        plt.savefig(check_filename, dpi=300)
        print('\nfigure saved: '+str(check_plot_name))
        plt.close(check_plot_name)
    
    # plot the different parts of the spring onto the wagon plot
    plt.figure(wagon_plot_name)
    plt.plot(x_start, y_flat, 'k')
    plt.plot(x_end, y_flat, 'k')
    plt.plot(x_coils, y_coils, 'k')
    
    return 0
    
#-----------------------------------------------------------------------------#
    
def animate_wagon(t, x, v, auto_play=False):
    '''
    This function takes in a time trace that describes the motion of a spring-
    mass system in one dimension and outputs a movie showing the evolution of
    the time trace and corresponding movement of the mass.
    
    Input:
      - t = time discretization of the time trace
      - x = location of the mass at corresponding time
    
    Output:
      - wagon_movie.mp4 = mp4 file with the animation
    '''
    from matplotlib import animation         # for specifying the writer
    import pylab as plt                      
    import webbrowser                        # for opening files automatically
    
    # plotting: USER INPUTS!
    wagon_width = 0.25                       # hardcoded wagon size, [m]
    plot_name = 'wagon_movie'
    movie_filename = plot_name+'.mp4'
    n_images = len(t)            # total number of images computed
    skip_images = 1000                   # images to skip between animation frames
    
    # instantiate the figure
    fig = plt.figure(plot_name)
    # rescale the figure window to fit both subplots
    xdim, ydim = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches(xdim, 1.2*ydim, forward=True)
    # things that will not be changing in the loop
    wagon_white_space = wagon_width
    trace_white_space = (max(x)-min(x))/4
    x_white_space = (max(x)-min(x))/4
    v_white_space = (max(v)-min(v))/4
    # list of all frames to be captured (skipping, if needed) plus final frame
    all_frames = list(range(0,n_images,skip_images+1))+[n_images-1]
    # plotting: capturing the movie
    writer = animation.writers['ffmpeg'](fps=11)
    with writer.saving(fig, movie_filename, 300):
        # initiate the frame counter
        frame = 0
        # cycle through the frames that need to be caputured
        for n in all_frames:
            # plot the wagon and the spring at the current position
            plt.subplot(2,1,1)
            plt.cla()
            wagon_position = x[n]
            plot_wagon(plot_name, wagon_position, wagon_width)
            plot_spring(plot_name, wagon_position, wagon_width/3.0)
            plt.xlim(0.0,max(x)+wagon_white_space)
            plt.ylim(-wagon_white_space,2.0*wagon_white_space)
            plt.xlabel('$x(t) \,,\, [\,m\,]$')
            plt.title('$t = '+str(round(t[n],2))+'s$')
            # plot the time trace
            plt.subplot(2,2,3)
            plt.plot(t[:n],x[:n],'b-')
            plt.xlim(0.0,max(t))
            plt.ylim(min(x)-trace_white_space, max(x)+trace_white_space)
            plt.xlabel('$t \,,\, [\,s\,]$')
            plt.ylabel('$x(t) \,,\, [\,m\,]$')
            # plot the phase portrait
            plt.subplot(2,2,4)
            plt.cla()
            plt.plot(x[:n],v[:n],'r-')
            plt.plot(x[n],v[n],'r.')
            plt.xlim(min(x)-x_white_space,max(x)+x_white_space)
            plt.ylim(min(v)-v_white_space,max(v)+v_white_space)
            plt.xlabel('$x(t) \,,\, [\,m\,]$')
            plt.ylabel('$v(t) \,,\, [\,m/s\,]$')
            #fig.subplots_adjust(hspace=.5,wspace=0.3)
            plt.tight_layout()            
            # progress monitor
            percent_done = float(n)*100.0/(n_images-1)
            print('capturing fig. '+plot_name+' (frame #'+str(frame)+'): ', \
                   percent_done,'%')
            # capture the current frame
            writer.grab_frame()
            # increment the frame counter
            frame += 1
        # grab the last frame again
        writer.grab_frame()
    # free memory used for the plot
    plt.close(plot_name)
    # print message to the screen
    print('\n\tmovie saved: '+str(movie_filename)+'\n')
    # start playing the movie once it has been saved
    if auto_play:
        webbrowser.open(movie_filename)

    return 0

#-----------------------------------------------------------------------------#
# compute the total number of time points
n_time_points = int((t_end-t_0)/del_t + 1)

# initialize list for position, velocity, and time
x = []
v = []
t = []

# append initial conditions to lists
x.append(x_0)
v.append(v_0)
t.append(t_0)

# explicit time marching, Euler's method
for n in range(1,n_time_points):
    
    # update and append velocity
    v_n = -del_t*(k/m)*(x[n-1]-x_eq) + v[n-1]
    v.append(v_n)
    
    # update and append position
    x_n = del_t*v[n-1] + x[n-1]
    x.append(x_n)
    
    # update and append time value (for plotting only)
    t_n = t[n-1] + del_t
    t.append(t_n)

# compute the theoretical values of angular velocity and period
omega = math.sqrt(k/m)          # angular velocity, [rad/s]
period = 2.0*math.pi/omega      # period of oscillation, [s]




# plot the time trace of the solution
time_trace_plot_name = 'time_trace'
time_trace_filename = time_trace_plot_name+'.png'
plt.figure(time_trace_plot_name)
plt.plot(t,x)
plt.xlabel('$t \,,\, [\,s\,]$', fontsize=14)
plt.ylabel('$x(t) \,,\, [\,m\,]$', fontsize=14)
plt.title('$\omega = ' + str(round(omega,2)) + \
          '\quad T = ' + str(round(period,2)) + \
          '\quad \Delta t = ' + str(del_t) + '$')
# save plot and close
plt.savefig(time_trace_filename, dpi=300)
print('\nfigure saved: '+time_trace_plot_name)
plt.close(time_trace_plot_name)
# open the saved image
webbrowser.open(time_trace_filename)

# calculate the eigenvalues for the given constants
lambda_1 = -1j*(k/m)**(1/2)
lambda_2 = -lambda_1

print('\n\teigenvalues of the homogeneous system:')
print('\n    lambda_1:\t', lambda_1)
print('    lambda_2:\t', lambda_2)




'''
# plot a wagon attached to a spring
wagon_width = 0.25              # [m]
wagon_position = 1.6            # [m]

wagon_plot_name = 'wagon_plot'
wagon_filename = wagon_plot_name+'.png'

plt.figure(wagon_plot_name)
plot_wagon(wagon_plot_name, wagon_position, wagon_width)
plot_spring(wagon_plot_name, wagon_position, wagon_width/3.0)

# save plot and close
plt.savefig(wagon_filename, dpi=300)
print('\nfigure saved: '+str(wagon_plot_name)+'\n')
plt.close(wagon_plot_name)
# open saved image
#webbrowser.open(wagon_filename)
'''

# animate the time-accurate solution
animate_wagon(t, x, v, auto_play=True)