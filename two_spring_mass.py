# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 19:24:38 2017

@author: Kedar
"""

from matplotlib import pyplot as plt
import math
import webbrowser     # for automatically opening files
import numpy as np
import cmath          # for complex roots
from dft_practice import my_dft

# turn off interactive mode, so that plot windows don't pop up
plt.ioff()

# spring #1 definition
l_1 = 2.0                           # equilibrium spring length, [m]
k_1 = 4900.0                        # spring constant, [N/m]

# wagon #1 mass
m_1 = 250.0                         # [kg]

# wagon width
wagon_1_width = 0.25                # [m]

# spring #2 definition
l_2 = 2.0                           # equilibrium spring length, [m]
k_2 = 4900.0                        # spring constant, [N/m]

# wagon #2 mass
m_2 = 250.0                         # [kg]

# wagon #2 width
wagon_2_width = wagon_1_width       # [m]

# initial displacement of wagon #2 
# (imagine taking the system set up above and then moving wagon #2 by the 
#  following distance, positive or negative. the new, initial position of 
#  wagon #1 is computed automatically when the variable x_1_0 is set to 
#  'equilibrium' below.)
delta_x_2 = 1.0                 # [m]

# intial conditions (think of the wagons as point masses here)
x_1_0 = 'equilibrium'                       # intial wagon #1 position, [N/m]
v_1_0 = 0.0                                 # intial wagon #1 velocity, [m/s]
x_2_0 = l_1+wagon_1_width+l_2+delta_x_2     # intial wagon #2 position, [N/m]
v_2_0 = 0.0                                 # intial wagon #2 velocity, [m/s]
t_0 = 0.0                                   # intial time, [s]

# time-step definition
del_t = 0.001              # time step, [s]
t_end = 10                  # final time, [s]

# if x_1 position is given as 'equilibrium', compute the position of wagon #1
# given the starting position of wagon #2
if x_1_0 == 'equilibrium':
    if not v_1_0==0.0:
        raise ValueError('cannot compute equilibrium position of wagon #1 ' + \
                         'if it is moving! set v_1_0 to zero!')
    elif not v_2_0==0.0:
        raise ValueError('cannot compute equilibrium position of wagon #1 ' + \
                         'if wagon #2 is moving! set v_2_0 to zero!')
    else:
        x_1_0 = l_1 + (k_2/(k_1+k_2))*delta_x_2

# create a dictionary defining the problem
my_two_spring_problem = {'m_1': m_1,            # mass #1, [kg]
                         'k_1': k_1,            # spring constant #1, [N/m] 
                         'l_1': l_1,            # equilibrium spring #1 length, [m]
                         'x_1_0': x_1_0,        # intial mass #1 position, [N/m]
                         'v_1_0': v_1_0,        # intial mass #1 velocity, [m/s]
                         'm_2': m_2,            # mass #2, [kg]
                         'k_2': k_2,            # spring constant #2, [N/m] 
                         'l_2': l_2,            # equilibrium spring #2 length, [m]
                         'x_2_0': x_2_0,        # intial mass #2 position, [N/m]
                         'v_2_0': v_2_0,        # intial mass #2 velocity, [m/s]
                         't_0': t_0}            # intial time, [s]


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
def plot_wagon(wagon_plot_name, x, wagon_width, wagon_color='brown'):
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
    plt.fill_between(x_horizontal,y_bottom,y_top,facecolor=wagon_color)
    plt.axis('equal')

#-----------------------------------------------------------------------------#    
def plot_spring(wagon_plot_name, starting_point, ending_point, flat_end_length, plot_func=False):
    '''
    This function plots a spring stretching from the starting point to the 
    ending point.
    '''
    import pylab as plt
    import math
    
    # check to make sure flat-end length is allowed
    if 2*flat_end_length >= ending_point-starting_point:
        raise ValueError('\n The flat-end length is too long! \n')
    
    # spring centerline height
    centerline_height = (flat_end_length*3.0/2.5)*(1/2+1/2.5)
    
    # spring characteristics
    n_coils = 10                            # no. of coils to be plotted
    coil_diameter = flat_end_length         # [m]
    
    # draw the starting and ending flat bits
    x_start = [starting_point, starting_point+flat_end_length]
    x_end = [ending_point-flat_end_length, ending_point]
    y_flat = [centerline_height]*2
    
    # distance to be spanned by coils
    coiled_span = ending_point-starting_point-2*flat_end_length
    
    # number of revolutions required to draw the desired number of coils
    coils_to_span = n_coils+0.5
    
    # value of the lifted cosine function at the number of coils required
    # N.B. this function has a period of one. the argument t is in units of
    # periods. func = 2*t - cos(2*pi*t) + 1
    func_at_span = 2*coils_to_span - math.cos(2*math.pi*coils_to_span) + 1
    
    # compute the scaling factor
    scaling_fac = coiled_span/func_at_span
    
    # draw the coils
    coil_points = 300
    delta_coil = coils_to_span/(coil_points-1)
    t_coils = [0]
    x_coils = [starting_point+flat_end_length]
    y_coils = [centerline_height]
    for i in range(coil_points):
        # time points
        t_coil = i*delta_coil
        t_coils.append(t_coil)
        # the values computed from the func are stretched to match the coiled-
        # span length and then added to the flat-end length
        x_coils.append(starting_point+flat_end_length + scaling_fac*(2.0*t_coil \
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
    
#-----------------------------------------------------------------------------#
def animate_wagon(t, x_1, x_2, v_1, v_2, wagon_1_width, wagon_2_width, auto_play=False):
    '''
    This function takes in a time trace that describes the motion of a spring-
    mass system in one dimension and outputs a movie showing the evolution of
    the time trace and corresponding movement of the mass.
    
    Input:
      - t = time discretization of the time trace
      - x = location of the mass at corresponding time
      - v = velocity of the mass at corresponding time
    
    Output:
      - wagon_movie.mp4 = mp4 file with the animation
    '''
    from matplotlib import animation         # for specifying the writer
    import pylab as plt                      
    import webbrowser                        # for opening files automatically
    
    # plotting: USER INPUTS!
    plot_name = 'wagons_movie'
    movie_filename = plot_name+'.mp4'
    n_images = len(t)            # total number of images computed
    skip_images = int(n_images/100.0)                   # images to skip between animation frames
    
    # instantiate the figure
    fig = plt.figure(plot_name)
    # rescale the figure window to fit both subplots
    xdim, ydim = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches(xdim, 1.2*ydim, forward=True)
    # things that will not be changing in the loop
    wagon_white_space = wagon_1_width
    trace_white_space = (max(x_2)-min(x_1))/4.0
    max_v = max(max(v_1),max(v_2))
    min_v = min(min(v_1),min(v_2))
    v_white_space = (max_v-min_v)/4
    max_x = max(max(x_1),max(x_2))
    min_x = min(min(x_1),min(x_2))
    x_white_space = (max_x-min_x)/4
    x_1_white_space = (max(x_1)-min(x_1))/4
    x_2_white_space = (max(x_2)-min(x_2))/4
    wagon_1_color = 'blue'
    wagon_2_color = 'red'
    plt.subplot(2,1,1)
    plt.xlim(0.0,max(x_2)+wagon_white_space)
    # select the desired axes of the phase portrait
    phase_axes = 'velocities-positions'
    phase_axes = 'positions'
    # list of all frames to be captured (skipping, if needed) plus final frame
    all_frames = [0]+list(range(0,n_images,skip_images+1))+[n_images-1]
    # plotting: capturing the movie
    writer = animation.writers['ffmpeg'](fps=int(len(all_frames)/(t[-1]-t[0])))
    with writer.saving(fig, movie_filename, 300):
        # initiate the frame counter
        frame = 0
        # cycle through the frames that need to be caputured
        for n in all_frames:
            # plot the wagons and the springs at the current position
            plt.subplot(2,1,1)
            plt.cla()
            wagon_1_position = x_1[n]
            plot_wagon(plot_name, wagon_1_position, wagon_1_width, wagon_color=wagon_1_color)
            plot_spring(plot_name, 0.0, wagon_1_position, wagon_1_width/3.0)
            wagon_2_position = x_2[n]
            plot_wagon(plot_name, wagon_2_position, wagon_2_width, wagon_color=wagon_2_color)
            plot_spring(plot_name, wagon_1_position+wagon_1_width, wagon_2_position, wagon_1_width/3.0)
            plt.xlim(0.0,max(x_2)+2.0*wagon_white_space)
            plt.ylim(-wagon_white_space,2.0*wagon_white_space)
            plt.xlabel('$x(t) \,,\, [\,m\,]$')
            plt.title('$t = '+str(round(t[n],2))+'s$')
            # plot the time trace
            plt.subplot(2,2,3)
            plt.cla()
            plt.plot(t[:n],x_1[:n],wagon_1_color)
            plt.plot(t[:n],x_2[:n],wagon_2_color)
            plt.xlim(0.0,max(t))
            plt.ylim(min(x_1)-trace_white_space, max(x_2)+trace_white_space)
            plt.xlabel('$t \,,\, [\,s\,]$')
            plt.ylabel('$x(t) \,,\, [\,m\,]$')
            # plot the phase portrait
            plt.subplot(2,2,4)
            plt.cla()
            if phase_axes == 'velocities-positions':
                plt.plot(x_1[:n],v_1[:n],wagon_1_color+'-')
                plt.plot(x_1[n],v_1[n],wagon_1_color+'.')
                plt.plot(x_2[:n],v_2[:n],wagon_2_color+'-')
                plt.plot(x_2[n],v_2[n],wagon_2_color+'.')
                plt.xlim(min_x-x_white_space,max_x+x_white_space)
                plt.ylim(min_v-v_white_space,max_v+v_white_space)
                plt.xlabel('$x(t) \,,\, [\,m\,]$')
                plt.ylabel('$v(t) \,,\, [\,m/s\,]$')
            if phase_axes == 'positions':
                plt.plot(x_1[:n],x_2[:n],'m-')
                plt.plot(x_1[n],x_2[n],'m.')
                plt.xlim(min(x_1)-x_1_white_space, max(x_1)+x_1_white_space)
                plt.ylim(min(x_2)-x_2_white_space, max(x_2)+x_2_white_space)
                plt.xlabel('$x_1(t) \,,\, [\,m\,]$')
                plt.ylabel('$x_2(t) \,,\, [\,m\,]$')
            #fig.subplots_adjust(hspace=.5)
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

#-----------------------------------------------------------------------------#
def make_plots(x_1, x_2, x_1_0, x_2_0, l_1, l_2, wagon_1_width, wagon_2_width, 
               auto_open=False):
    '''
    make plots of the equilibrium setup, the displacement of the second mass,
    and the initial condition
    '''
    # set the white spaces
    wagon_white_space = wagon_1_width
    trace_white_space = (max(x_2)-min(x_1))/4.0
    max_v = max(max(v_1),max(v_2))
    min_v = min(min(v_1),min(v_2))
    v_white_space = (max_v-min_v)/4
    max_x = max(max(x_1),max(x_2))
    min_x = min(min(x_1),min(x_2))
    x_white_space = (max_x-min_x)/4
    x_1_white_space = (max(x_1)-min(x_1))/4
    x_2_white_space = (max(x_2)-min(x_2))/4
    # set the wagon colors
    wagon_1_color = 'blue'
    wagon_2_color = 'red'
    
    # plot the equilibrium setup
    equilibrium_plot_name = 'equilibrium_setup'
    equilibrium_filename = equilibrium_plot_name+'.png'
    # instantiate the figure
    plt.figure(equilibrium_plot_name)
    # rescale the figure window to fit both subplots
    xdim, ydim = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches(2.0*xdim, ydim, forward=True)
    # plot the equilibrium setup
    #plt.subplot(1,2,1)
    plt.subplot(3,2,3)
    plot_wagon(equilibrium_plot_name, l_1, wagon_1_width, wagon_color=wagon_1_color)
    plot_spring(equilibrium_plot_name, 0.0, l_1, wagon_1_width/3.0)
    plot_wagon(equilibrium_plot_name, l_1+wagon_1_width+l_2, wagon_2_width, wagon_color=wagon_2_color)
    plot_spring(equilibrium_plot_name, l_1+wagon_1_width, l_1+wagon_1_width+l_2, wagon_1_width/3.0)
    plt.xlim(0.0,max(x_2)+2.0*wagon_white_space)
    plt.ylim(-wagon_white_space,2.0*wagon_white_space)
    plt.xlabel('$x(t) \,,\, [\,m\,]$')
    # plot the equilibrium positions on the phase portrait
    plt.subplot(1,2,2)
    plt.plot(l_1,l_1+wagon_1_width+l_2,'mo')
    plt.xlim(min(x_1)-x_1_white_space, max(x_1)+x_1_white_space)
    plt.ylim(min(x_2)-x_2_white_space, max(x_2)+x_2_white_space)
    plt.xlabel('$x_1(t) \,,\, [\,m\,]$')
    plt.ylabel('$x_2(t) \,,\, [\,m\,]$')
    plt.title('$x_1='+str(l_1)+'\,m\quad,\quad x_2='+str(l_1+wagon_1_width+l_2)+'\,m$')
    # save plot and close
    plt.savefig(equilibrium_filename, dpi=300)
    print('\nfigure saved: '+equilibrium_plot_name)
    plt.close(equilibrium_plot_name)
    # open the saved image
    if auto_open:
        webbrowser.open(equilibrium_filename)
    
    # plot the forced displacement of the second mass
    m2_displaced_plot_name = 'm2_displaced'
    m2_displaced_filename = m2_displaced_plot_name+'.png'
    # instantiate the figure
    plt.figure(m2_displaced_plot_name)
    # rescale the figure window to fit both subplots
    xdim, ydim = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches(2.0*xdim, ydim, forward=True)
    # plot the setup with the second wagon displaced
    #plt.subplot(1,2,1)
    plt.subplot(3,2,3)
    plot_wagon(m2_displaced_plot_name, l_1, wagon_1_width, wagon_color=wagon_1_color)
    plot_spring(m2_displaced_plot_name, 0.0, l_1, wagon_1_width/3.0)
    plot_wagon(m2_displaced_plot_name, x_2_0, wagon_2_width, wagon_color=wagon_2_color)
    plot_spring(m2_displaced_plot_name, l_1+wagon_1_width, x_2_0, wagon_1_width/3.0)
    plt.xlim(0.0,max(x_2)+2.0*wagon_white_space)
    plt.ylim(-wagon_white_space,2.0*wagon_white_space)
    plt.xlabel('$x(t) \,,\, [\,m\,]$')
    # plot the displaced positions on the phase portrait
    plt.subplot(1,2,2)
    plt.plot(l_1,l_1+wagon_1_width+l_2,'m.')
    plt.plot(l_1,x_2_0,'mo')
    plt.xlim(min(x_1)-x_1_white_space, max(x_1)+x_1_white_space)
    plt.ylim(min(x_2)-x_2_white_space, max(x_2)+x_2_white_space)
    plt.xlabel('$x_1(t) \,,\, [\,m\,]$')
    plt.ylabel('$x_2(t) \,,\, [\,m\,]$')
    plt.title('$x_1='+str(l_1)+'\,m\quad,\quad x_2='+str(x_2_0)+'\,m$')
    # save plot and close
    plt.savefig(m2_displaced_filename, dpi=300)
    print('\nfigure saved: '+m2_displaced_plot_name)
    plt.close(m2_displaced_plot_name)
    # open the saved image
    if auto_open:
        webbrowser.open(m2_displaced_filename)
    
    # plot the intial condition
    initial_condition_plot_name = 'initial_condition'
    initial_condition_filename = initial_condition_plot_name+'.png'
    # instantiate the figure
    plt.figure(initial_condition_plot_name)
    # rescale the figure window to fit both subplots
    xdim, ydim = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches(2.0*xdim, ydim, forward=True)
    # plot the initial condition of the setup
    #plt.subplot(1,2,1)
    plt.subplot(3,2,3)
    plot_wagon(initial_condition_plot_name, x_1_0, wagon_1_width, wagon_color=wagon_1_color)
    plot_spring(initial_condition_plot_name, 0.0, x_1_0, wagon_1_width/3.0)
    plot_wagon(initial_condition_plot_name, x_2_0, wagon_2_width, wagon_color=wagon_2_color)
    plot_spring(initial_condition_plot_name, x_1_0+wagon_1_width, x_2_0, wagon_1_width/3.0)
    plt.xlim(0.0,max(x_2)+2.0*wagon_white_space)
    plt.ylim(-wagon_white_space,2.0*wagon_white_space)
    plt.xlabel('$x(t) \,,\, [\,m\,]$')
    # plot the initial-condition positions on the phase portrait
    plt.subplot(1,2,2)
    plt.plot(l_1,l_1+wagon_1_width+l_2,'m.')
    plt.plot(l_1,x_2_0,'m.')
    plt.plot(x_1_0,x_2_0,'mo')
    plt.xlim(min(x_1)-x_1_white_space, max(x_1)+x_1_white_space)
    plt.ylim(min(x_2)-x_2_white_space, max(x_2)+x_2_white_space)
    plt.xlabel('$x_1(t) \,,\, [\,m\,]$')
    plt.ylabel('$x_2(t) \,,\, [\,m\,]$')
    plt.title('$x_1='+str(x_1_0)+'\,m\quad,\quad x_2='+str(x_2_0)+'\,m$')
    # save plot and close
    plt.savefig(initial_condition_filename, dpi=300)
    print('\nfigure saved: '+initial_condition_plot_name)
    plt.close(initial_condition_plot_name)
    # open the saved image
    if auto_open:
        webbrowser.open(initial_condition_filename)
    
    # plot the time trace of the solution
    time_trace_plot_name = 'time_traces'
    time_trace_filename = time_trace_plot_name+'.png'
    plt.figure(time_trace_plot_name)
    plt.plot(t,x_1,wagon_1_color,label='$m_1$')
    plt.plot(t,x_2,wagon_2_color,label='$m_2$')
    plt.xlabel('$t \,,\, [\,s\,]$', fontsize=14)
    plt.ylabel('$x(t) \,,\, [\,m\,]$', fontsize=14)
    plt.xlim(0.0,max(t))
    plt.ylim(min(x_1)-trace_white_space, max(x_2)+trace_white_space)
    plt.legend(loc='best')
    # save plot and close
    plt.savefig(time_trace_filename, dpi=300)
    print('\nfigure saved: '+time_trace_plot_name)
    plt.close(time_trace_plot_name)
    # open the saved image
    if auto_open:
        webbrowser.open(time_trace_filename)
        
#-----------------------------------------------------------------------------#

# compute the total number of time points
n_time_points = int((t_end-t_0)/del_t + 1)

# initialize list for positions, velocities, and time
x_1 = []
v_1 = []
x_2 = []
v_2 = []
x = []
v = []
t = []

# append initial conditions to lists
x_1.append(x_1_0)
v_1.append(v_1_0)
x_2.append(x_2_0)
v_2.append(v_2_0)
t.append(t_0)

# make the relevant vectors and matrices
x.append(np.array([[x_1_0], 
                   [x_2_0]]))
v.append(np.array([[v_1_0],
                   [v_2_0]]))
K = np.array([[ -(k_1+k_2)/m_1   ,       k_2/m_1   ],
              [   k_2/m_2        ,      -k_2/m_2    ]])
l = np.array([[(k_1*l_1 - k_2*(wagon_1_width+l_2))/m_1],
              [k_2*(wagon_1_width+l_2)/m_2]])
                 
# explicit time marching, Euler's method
for n in range(1,n_time_points):
    
    # update and append velocity vector
    v_n = v[n-1] + del_t*(np.dot(K,x[n-1]) + l)
    v.append(v_n)
    # append the individual positions
    v_1.append(v_n[0][0])
    v_2.append(v_n[1][0])
    
    # update and append position vector
    x_n = x[n-1] + del_t*v[n-1]
    x.append(x_n)
    # append the individual positions
    x_1.append(x_n[0][0])
    x_2.append(x_n[1][0])
    
    # update and append time value (for plotting only)
    t_n = t[n-1] + del_t
    t.append(t_n)

# calculate the eigenvalues of the homogeneous system
a = (m_1*m_2)**2
b = m_1*m_2*(k_1*m_2 + k_2*(m_1+m_2))
c = k_1*k_2*m_1*m_2
lambda_1 = -cmath.sqrt((-b - cmath.sqrt(b**2-4.0*a*c))/(2.0*a))
lambda_2 = cmath.sqrt((-b - cmath.sqrt(b**2-4.0*a*c))/(2.0*a))
lambda_3 = -cmath.sqrt((-b + cmath.sqrt(b**2-4.0*a*c))/(2.0*a))
lambda_4 = cmath.sqrt((-b + cmath.sqrt(b**2-4.0*a*c))/(2.0*a))

# plot the initial conditions, solutions, and phase portraits
make_plots(x_1, x_2, x_1_0, x_2_0, l_1, l_2, wagon_1_width, wagon_2_width, auto_open=True)

# animate the time-accurate solution
animate_wagon(t, x_1, x_2, v_1, v_2, wagon_1_width, wagon_2_width, auto_play=True)


# take the DFT of both time traces
s, F, powers, x_1_peaks_found, x_1_peak_boundaries = my_dft(t, x_1, 
                                                         percent_energy_AC_peaks=97,
                                                         shift_frequencies=True,
                                                         use_angular_frequencies=True,
                                                         plot_spectrum=True, 
                                                         plot_log_scale=True,
                                                         refine_peaks=True,
                                                         auto_open_plot=True,
                                                         verbose=True,
                                                         title_suffix='',
                                                         plot_suffix=' - x1',
                                                         use_fft=True)
                                                         
s, F, powers, x_2_peaks_found, x_2_peak_boundaries = my_dft(t, x_2, 
                                                         percent_energy_AC_peaks=97,
                                                         shift_frequencies=True,
                                                         use_angular_frequencies=True,
                                                         plot_spectrum=True, 
                                                         plot_log_scale=True,
                                                         refine_peaks=True,
                                                         auto_open_plot=True,
                                                         verbose=True,
                                                         title_suffix='',
                                                         plot_suffix=' - x2',
                                                         use_fft=True)

print('\neigenvalues of the homogeneous system:')
print('\n    lambda_1:\t', lambda_1)
print('    lambda_2:\t', lambda_2)
print('    lambda_3:\t', lambda_3)
print('    lambda_4:\t', lambda_4)
print('\n    N.B. (1) the four eigenvalues are two complex-conjugate pairs.')
print('         (2) the magnitudes of these eigenvalues represent the \n' + \
      '             angular frequencies at which the wagons oscillate')