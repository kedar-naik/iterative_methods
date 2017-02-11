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
delta_x_2 = 0.5                 # [m]

# intial conditions (think of the wagons as point masses here)
x_1_0 = 'equilibrium'                       # intial wagon #1 position, [N/m]
v_1_0 = 0.0                                 # intial wagon #1 velocity, [m/s]
x_2_0 = l_1+wagon_1_width+l_2+delta_x_2     # intial wagon #2 position, [N/m]
v_2_0 = 0.0                                 # intial wagon #2 velocity, [m/s]
t_0 = 0.0                                   # intial time, [s]

# time-step definition
del_t = 0.00005              # time step, [s]
t_end = 13.25                  # final time, [s]

# initialize the static forces required to hold the initial condition in place
f_1_0 = 0.0                 # [N]
f_2_0 = 0.0                 # [N]

# if x_1 position is given as 'equilibrium', compute the static position of 
# wagon #1 given the starting position of wagon #2. also compute the force on
# wagon #2 required to hold the system in the position of the initial condition 
if x_1_0 == 'equilibrium':
    if not v_1_0==0.0:
        raise ValueError('cannot compute equilibrium position of wagon #1 ' + \
                         'if it is moving! set v_1_0 to zero!')
    elif not v_2_0==0.0:
        raise ValueError('cannot compute equilibrium position of wagon #1 ' + \
                         'if wagon #2 is moving! set v_2_0 to zero!')
    else:
        x_1_0 = l_1 + (k_2/(k_1+k_2))*delta_x_2
        f_2_0 = (k_1*k_2/(k_1+k_2))*delta_x_2

# create a dictionary defining the problem
my_two_spring_problem = {'m_1': m_1,            # mass #1, [kg]
                         'k_1': k_1,            # spring constant #1, [N/m] 
                         'l_1': l_1,            # equilibrium spring #1 length, [m]
                         'm_2': m_2,            # mass #2, [kg]
                         'k_2': k_2,            # spring constant #2, [N/m] 
                         'l_2': l_2,            # equilibrium spring #2 length, [m]
                         'wagon_1_width': wagon_1_width}
# create a dictionary of initial conditions
initial_conditions = {'x_1_0': x_1_0,        # intial mass #1 position, [m]
                      'v_1_0': v_1_0,        # intial mass #1 velocity, [m/s]
                      'f_1_0': f_1_0,        # force required to hold mass #1 [N]
                      'x_2_0': x_2_0,        # intial mass #2 position, [m]
                      'v_2_0': v_2_0,        # intial mass #2 velocity, [m/s]
                      'f_2_0': f_2_0,        # force required to hold mass #2 [N]
                      't_0':   t_0}          # intial time, [s]

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
    skip_images = int(n_images/(5*(t[-1]-t[0])))                   # images to skip between animation frames
    
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
    # print message to the screen
    print('\nmaking a movie...quiet on the set!\n')
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
            print('\tcapturing fig. '+plot_name+' (frame #'+str(frame)+'): ', \
                   round(percent_done,2),'%')
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
def make_plots(t, x_1, x_2, x_1_0, x_2_0, l_1, l_2, wagon_1_width, wagon_2_width, 
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
class two_spring_mass_ode_system:
    '''
    class for defining and evaluating the system of odes governing the two-
    spring-two-mass problem
    '''
    # class constructor (accepts the actual omegas from the user)
    def __init__(self, two_spring_problem_dict, initial_conditions_dict):
        # extract the relevant values from the dictionary
        self.m_1 = two_spring_problem_dict['m_1']   # mass #1, [kg]
        self.m_2 = two_spring_problem_dict['m_2']   # mass #2, [kg]
        self.l_1 = two_spring_problem_dict['l_1']   # equilibrium spring #1 length, [m]
        self.k_1 = two_spring_problem_dict['k_1']   # spring constant #1, [N/m]
        self.k_2 = two_spring_problem_dict['k_2']   # spring constant #2, [N/m]
        self.l_2 = two_spring_problem_dict['l_2']   # equilibrium spring #2 length, [m]
        self.wagon_1_width = two_spring_problem_dict['wagon_1_width']
        # compute the constant "stiffness" matrix
        self.K = np.array([[   -(self.k_1+self.k_2)/self.m_1   ,       self.k_2/self.m_1   ],
                           [     self.k_2/self.m_2        ,      -self.k_2/self.m_2   ]])
        # compute the constant "offset" vector
        self.l = np.array([[(self.k_1*self.l_1 - self.k_2*(self.wagon_1_width+self.l_2))/self.m_1],
                           [      self.k_2*(self.wagon_1_width+self.l_2)/self.m_2      ]])
        # extract the forces required to hold the initial condition in place
        self.f_0 = []
        self.f_0.append(initial_conditions_dict['f_1_0'])
        self.f_0.append(initial_conditions_dict['f_2_0'])
        # create another small list for the two masses (for convenience later)
        self.m = [self.m_1, self.m_2]
        # extract the time at which the simulation begins
        self.t_0 = initial_conditions_dict['t_0']
        
    # function to evaluate the expression that follows x" = ...
    def evaluate(self, x):
        '''
        given a vector of positions [x_1, x_2]^T, return the acceleration 
        vector [x_1", x_2"]^T
        '''
        import numpy as np
        # compute the acceleration vector
        x_double_dot = np.dot(self.K,x) + self.l
        # return the acceleration vector
        return x_double_dot
    
    # function to evaluate the expression that follows x_i" = ...
    def evaluate_RHS_i(self, mass_number, f, t, static_initial_condition=False):
        '''
        given values of x_1 and x_2, this function returns the RHS of the
        equation of motion for the i-th cart. it is okay if x_1 and x_2 are
        numpy arrays.
        Input:
            - mass_number: mass whose motion is needed (either wagon #1 or #2)
            - f: current solution, as a list of numpy column arrays
                  [ [[x_1(t_1)],      [[x_2(t_1)],
                     [x_1(t_2)],       [x_2(t_2)],
                        ...,       ,      ..., 
                     [x_1(t_N)]]       [x_2(t_N)]] ]
            - t: current time(s) as a numpy array [[t_1],[t_2],...,[t_N]]
            - static_initial_condition: set this to True if computing the first 
                                        step of a time-accurate problem in 
                                        which the initial condition is static.
                                        set it to false when computing a 
                                        harmonic-balance solution. why? because 
                                        from the point of view of the harmonic-
                                        balance, the simulation begins at 0+ 
                                        (immediately after the static forces
                                        holding the initial coniditon in place 
                                        have been released).
        Ouput:
            - x_i_double_dot: acceleration(s) of the given mass
        '''
        # array index correspoding to the mass number
        mass_index = mass_number-1
        # pull out the arrays corresponding to each variable
        x_i_double_dot = np.zeros_like(f[0])
        for m in range(len(f)):
            x_i_double_dot += self.K[mass_index][m]*f[m]
        # add the constant numpy array at the end of this equation of motion
        x_i_double_dot += self.l[mass_index]*np.ones_like(f[0])
        # include the static force required to hold the initial condition
        if static_initial_condition and self.t_0 in t:
            # find the index corresponding to the initial time
            t_0_index = list(t).index(self.t_0)
            # run through the solution vector and include the initial condition
            for j in range(len(t)):
                if j == t_0_index:
                    # if dealing with a static position (that is, to be used in
                    # the time-accurate simulation at t=0)
                    x_i_double_dot[j] += self.f_0[mass_index]/self.m[mass_index]
        # return this value
        return x_i_double_dot
    
    # function to analytically compute any of the jacobians
    def evaluate_dRi_dxj(self, i, j, D_HB2):
        '''
        evaluates the analytical expression for the jacobians
        '''
        I = np.eye(len(D_HB2))
        if i == j:
            dRi_over_dxj = self.K[i,j]*I-D_HB2
        else:
            dRi_over_dxj = self.K[i,j]*I
        return dRi_over_dxj
    
    # if x1 is set to be periodic, then x2 is...
    def x2_when_periodic_x1(self, f_HB, t_HB, omegas, curve_type):
        import numpy as np
        if curve_type == 'bump':
            x_2 = (1.0/self.k_2)*(self.m_1*(self.k_1+self.k_2)*f_HB[0][0] + self.k_2*(self.wagon_1_width+self.l_2) - self.k_1*self.l_1) + (self.m_1/(2.0*self.k_2))*((max(f_HB[0])-min(f_HB[0]))*(self.k_1+self.k_2+(omegas[0]**2-self.k_1-self.k_2)*np.cos(omegas[0]*t_HB)))
        if curve_type == 'valley':
            x_2 = (1.0/self.k_2)*(self.m_1*(self.k_1+self.k_2)*f_HB[0][0] - self.k_2*(self.wagon_1_width+self.l_2) - self.k_1*self.l_1) + (self.m_1/(2.0*self.k_2))*((max(f_HB[0])-min(f_HB[0]))*(self.k_1+self.k_2+(omegas[0]**2-self.k_1-self.k_2)*np.cos(omegas[0]*t_HB)))
        return x_2
        
    # if x2 is set to be periodic, then x1 is...
    def x1_when_periodic_x2(self, f_HB, t_HB, omegas, curve_type):
        import numpy as np
        if curve_type == 'bump':
            x_1 = f_HB[1][0] - self.wagon_1_width - self.l_1 + (1/2)*(max(f_HB[1])-min(f_HB[1]))*(1 + (1+self.m_2*omegas[0]**2/self.k_2)*np.cos(omegas[0]*t_HB))
        if curve_type == 'valley':
            x_1 = f_HB[1][0] - self.wagon_1_width - self.l_1 - (1/2)*(max(f_HB[1])-min(f_HB[1]))*(1 + (1+self.m_2*omegas[0]**2/self.k_2)*np.cos(omegas[0]*t_HB))
        return x_1
        
#-----------------------------------------------------------------------------#
def solve_HB_problem(omegas, time_discretization, the_ode, delta_tau, 
                     constant_init_guess, residual_convergence_criteria, 
                     make_plot=False, auto_open_plot=False, make_movie=False, 
                     auto_play_movie=False, verbose=True, spring_mass=False):
    '''
    this subroutine returns a convergered harmonic-balance solution comprising
    the K angular frequencies given.
    Inputs:
        - omegas: list of K angular frequencies to build up the HB solution
        - time_discretization: 'use_T1' (equally spaced points just spanning 
                                the period corresponding to the lowest omega)
                                                  OR
                               'use_Nyquist' (use the time interval needed to 
                                just capture bandwidth, i.e. highest omega)
        - the_ode: instance of governing_ode class that define the ODE
        - constant_init_guess: constant-value guess for the HB initial solution
        - delta_tau: pseudo-time step size
        - residual_convergence_criteria: desired residual convergence criteria
                                                        OR
                                         'fully converged' which sets the 
                                         criterion three orders of magnitude
                                         above machine zero
        - make_plot: plot the converged solution and the residual history
        - auto_open: automatically open the plot
        - make_movie: animate the convergence process
        - auto_play: automatically open and start playing the movie
        - verbose: print residual convergence history to the screen
        - spring_mass: if running a spring-mass case, supply initial condition 
                       in place of constant_init_guess as a numpy array
    Output:
        - t_HB: the time instances over which the HB solution is defined
        - f_HB: the converged harmonic-balance solution
    '''
    import numpy as np
    import math
    from matplotlib import pyplot as plt
    from matplotlib import animation         # for specifying the writer
    import sys
    from HB_practice import HB_omega_check, harmonic_balance_operator
    from HB_practice import fourierInterp_given_freqs
    
    # if the residual convergence criterion is set to 'fully converged', then
    # set it to the right value by using machine zero
    if residual_convergence_criteria == 'fully converged':
        machine_zero = sys.float_info.epsilon
        fully_converged = 1e3*machine_zero
        residual_convergence_criteria = fully_converged
    # if using the T1 time discretization, check to see if all the given 
    # angular frequencies are valid, if not, print a warning to the screen
    omega_dicts, invalid_omegas = HB_omega_check(omegas, make_plots=False)
    # if there are invalid frequencies, print them to the screen
    if invalid_omegas:
        if len(invalid_omegas) == 1:
            print('\n\tthere is an inadmissible angular frequency!')
        else:
            print('\n\tthere are inadmissible angular frequencies!')
        print('\tconsider using Nyquist-based discretization!')
        for omega_tuple in invalid_omegas:
            bad_omega_number = omega_tuple[0]
            bad_omega_value = omega_tuple[1]
            print('\n\t\t- omega #' + str(bad_omega_number) + ' (' + \
                  str(bad_omega_value) + ' rad/sec) is inadmissible!')
    # maximum number of pseudo-time steps to try (can be changed, if needed)
    max_pseudo_steps = 5
    # print message to the screen
    if verbose:
        print('computing the harmonic-balance solution...')
    else:
        print('\tcomputing the harmonic-balance solution...', end ='')
    # create the harmonic balance operator matrix and find the time instances
    D_HB, t_HB = harmonic_balance_operator(omegas, time_discretization)
    # extract the number of time instances
    N = len(t_HB)
    if spring_mass:
        # multiply the operator matrix by itself (since we need 2nd derivatives)
        D_HB2 = np.dot(D_HB,D_HB)
        

        # rename the initial condition provided
        x_0 = constant_init_guess
        # figure out the number of masses (d.o.f.) we have
        n_masses = len(x_0)
        
        
        # exact solutions for x_1 and x_2 extracted from the time-accurate
        # simulation at the 2*K+1 harmonic-balance time instances
        f_HB = [np.array([[2.25],[2.13332348],[1.7205478],[1.80193313],[2.05404768]]),
                np.array([[4.75],[4.37016178],[3.8913272],[3.84024355],[4.42041497]])]
        
        
        '''
        # determine how to pick the initial guess for the HB solution 
        # 'as constant' = replicate the initial condition across time instances
        # 'randomly' = guesses at the time instances are normally distributed 
        #              about the initial condition for that mass
        set_initial_guess = 'as constant'
        set_initial_guess = 'randomly'
        
        # create an initial HB guess
        f_HB = []
        for m in range(n_masses):
            if set_initial_guess == 'as constant':
                # copy the initial condition across N-1 instances
                f_HB.append(np.array([x_0[m] for i in range(N)]).reshape((N,1)))
            if set_initial_guess == 'randomly':
                # intitalize as normally distributed about the initial condition
                f_HB.append(x_0[m]+(10/1)*np.random.randn(N).reshape((N,1)))
                # set the values at the first time instance to the initial condtion
                f_HB[m][0] = x_0[m]
        '''
        
        
        # "really close to the actual solution!"
        f_HB = [np.array([[2.25],[2.0],[1.7],[1.8],[2.0]]),
                np.array([[4.75],[4.4],[3.4],[3.8],[4.4]])]
        
        
        # at each pseudo-time iteration, construct positive and negative 
        # sinusoidal curves and see if either yields a smaller residual. this
        # might help the system jump out of local minima (a creationist's 
        # genetic algorithm)
        try_periodic_solutions = True

        # create a list for the residual evolution history
        residual_history = [[] for mass in range(n_masses)]
        
        
    else:
        # create an initial guess for the HB solution
        f_HB = np.array([constant_init_guess]*len(t_HB)).reshape((len(t_HB),1))
        #f_HB = init_guess+np.random.rand(len(t_HB)).reshape((len(t_HB),1))
        
        # create a list for the residual evolution history
        residual_history = []
   
 
    # create a list for the solution-evolution history
    f_HB_history = [np.copy(f_HB)]
    
    # pick the type of pseudo-time stepping scheme ('explicit' or 'implicit')
    pseudotimestepping_scheme = 'explicit'
    pseudotimestepping_scheme = 'implicit'
    
    # dynamically adjust the pseudo-time step size
    adjust_delta_tau = False
    delta_tau_init = delta_tau
    # record pseudo-time step
    delta_tau_hist = []
    delta_tau_hist.append(delta_tau)
    going_up_count = 0

    
    # start the pseudo-transient continuation method
    for iteration in range(max_pseudo_steps):
        
        # compute the residual vector corresponding the current solution
        if spring_mass:
            residual = []
            norm_residual = []
            f_HB_plus = [np.zeros((N,1)) for mass in range(n_masses)]
            f_HB_minus = [np.zeros((N,1)) for mass in range(n_masses)]
            
            for m in range(n_masses):
                
                func_evaluations = the_ode.evaluate_RHS_i(m+1, f_HB, t_HB)
                matrix_vector_product = np.dot(D_HB2,f_HB[m])
                residual.append(func_evaluations - matrix_vector_product)
                norm_residual.append(np.linalg.norm(residual[m]))
                residual_history[m].append(norm_residual[m])
                
                if try_periodic_solutions:
                    
                    # turn the time instances into a column vector
                    t_HB = t_HB.reshape(N,1)
                    # construct a periodic bump at the time instances for the 
                    # m-th mass's solution. leave the other mass's solution alone
                    f_HB_plus[m] = f_HB[m][0] + (1/2)*(max(f_HB[m])-min(f_HB[m]))*(1.0-np.cos(omegas[0]*t_HB))
                    if m == 0:
                        # x1 is periodic, set corresponding x2
                        f_HB_plus[1-m] = the_ode.x2_when_periodic_x1(f_HB, t_HB, omegas, curve_type='bump')
                    if m == 1:
                        # x2 is periodic, set corresponding x1
                        f_HB_plus[1-m] = the_ode.x1_when_periodic_x2(f_HB, t_HB, omegas, curve_type='bump')
                    #f_HB_plus[1-m] = f_HB[1-m]
                    
                    # check the resulting residual
                    func_evaluations = the_ode.evaluate_RHS_i(m+1, f_HB_plus, t_HB)
                    matrix_vector_product = np.dot(D_HB2,f_HB_plus[m])
                    residual_plus = func_evaluations - matrix_vector_product
                    norm_residual_plus = np.linalg.norm(residual_plus)
                    
                    # construct a periodic valley at the time instances
                    f_HB_minus[m] = f_HB[m][0] - (1/2)*(max(f_HB[m])-min(f_HB[m]))*(1.0-np.cos(omegas[0]*t_HB))
                    if m == 0:
                        # x1 is periodic, set corresponding x2
                        f_HB_plus[1-m] = the_ode.x2_when_periodic_x1(f_HB, t_HB, omegas, curve_type='valley')
                    if m == 1:
                        # x2 is periodic, set corresponding x1
                        f_HB_plus[1-m] = the_ode.x1_when_periodic_x2(f_HB, t_HB, omegas, curve_type='valley')
                    #f_HB_minus[1-m] = f_HB[1-m]
                    # check the resulting residual
                    func_evaluations = the_ode.evaluate_RHS_i(m+1, f_HB_minus, t_HB)
                    matrix_vector_product = np.dot(D_HB2,f_HB_minus[m])
                    residual_minus = func_evaluations - matrix_vector_product
                    norm_residual_minus = np.linalg.norm(residual_minus)
                    
                    # put all three residual norms into an array
                    candidate_norms = [norm_residual[m], norm_residual_plus, norm_residual_minus]
                    # find the index of the smallest candidate residual norms
                    min_index = candidate_norms.index(min(candidate_norms))
                    
                    
                    
                    # reassign the current solution and corresponding variables
                    if min_index == 1:
                        # f_HB_plus yields the smallest residual
                        #print('\n\n\n\n\n\n\tusing periodic bump!')
                        #print('\nmin_index =', min_index)
                        f_HB[m] = f_HB_plus[m]
                        residual[-1] = residual_plus
                        residual_history[m][-1] = norm_residual_plus
                    if min_index == 2:
                        # f_HB_minus yields the smallest residual
                        #print('\n\n\n\n\n\n\n\tusing periodic valley!')
                        #print('\nmin_index =', min_index)
                        f_HB[m] = f_HB_minus[m]
                        residual[-1] = residual_minus
                        residual_history[m][-1] = norm_residual_minus
                    
                        
                    
                    
                    
                    
                '''
                print('\nt_HB =', t_HB)
                print('\nf_HB_'+str(m+1)+' = ', f_HB[m])
                print('\nfunc_evaluations_'+str(m+1)+' = ', func_evaluations)
                print('\nmatrix_vector_product_'+str(m+1)+' = ', matrix_vector_product)
                print('\nresidual_'+str(m+1)+' = ', residual[m])                
                '''


                if verbose:    
                    if m == 0:
                        print('\n\titer: '+str(iteration)+'\t||residual_'+str(m+1)+'||: '+str(norm_residual[m]), end='')
                    else:
                        print('\t||residual_'+str(m+1)+'||: '+str(norm_residual[m]), end='')
                    
            #print('norm of sum =', np.linalg.norm(residual[0]+residual[1]))
            

        else:
            func_evaluations = the_ode.evaluate(t_HB,f_HB)
            matrix_vector_product = np.dot(D_HB,f_HB)
            residual = func_evaluations - matrix_vector_product
            norm_residual = np.linalg.norm(residual)
            residual_history.append(norm_residual)
            if verbose:
                print('\n\titer: '+str(iteration)+'\t||residual||: '+str(norm_residual), end='')
        
        # compute the "error," which is a function of the residual, for pseudo-
        # trasient continuation (pseudo-time stepping)
        I = np.eye(2*len(omegas)+1)     # identity matrix
        step_size = delta_tau
        if pseudotimestepping_scheme == 'explicit':
            B = step_size*I
            if spring_mass:
                # initialize a list of numpy arrays to hold the errors
                error = []
                for m in range(n_masses):
                    # compute the error vector for this variable
                    error.append(np.dot(B,residual[m]))
                    # set the error of the first time instance
                    #error[m][0] = 0.0
                    
                    #print('\nerror_'+str(m+1)+' = ', error[m])
            else:
                error = np.dot(B,residual)

        if pseudotimestepping_scheme == 'implicit':
            if spring_mass:
                
                # compute and store the Jacobians
                error = []
                for m in range(n_masses):
                    J = the_ode.K[m,m]*I - D_HB2
                    B = step_size*np.linalg.inv(I - step_size*J)
                    error.append(np.dot(B,residual[m]))
                    # set the error of the first time instance
                    error[m][0] = 0.0
                    
                    #print('\nerror_'+str(m+1)+' = ', error[m])
                    
                    
                '''
                # directly from gradient descent...
                error=[]
                for m in range(n_masses):
                    error.append(-step_size*np.dot(((the_ode.K[m,0]+the_ode.K[m,1])*I-D_HB2),np.ones(N).reshape(N,1)))
                    error[m][0] = 0.0
                '''

                '''
                # from 2D taylor series...
                error = [0.0, 0.0]
                dR1dx1 = the_ode.evaluate_dRi_dxj(0,0,D_HB2)
                dR1dx2 = the_ode.evaluate_dRi_dxj(0,1,D_HB2)
                dR2dx1 = the_ode.evaluate_dRi_dxj(1,0,D_HB2)
                dR2dx2 = the_ode.evaluate_dRi_dxj(1,1,D_HB2)
                # delta_x2
                error[1] = step_size*np.dot(np.linalg.inv(I-step_size*(dR2dx2+step_size*np.dot(np.dot(dR2dx1,np.linalg.inv(I-step_size*dR1dx1)),dR1dx2))),(residual[1]+step_size*np.dot(np.dot(dR2dx1,np.linalg.inv(I-step_size*dR1dx1)),residual[0])))
                # delta_x1
                error[0] = step_size*np.dot(np.linalg.inv(I-step_size*dR1dx1),residual[0]+np.dot(dR1dx2,error[1]))
                error[0][0] = 0.0
                error[1][0] = 0.0
                '''
                
        # if there's more than one mass, then use the max ||residual|| value
        if spring_mass:
            norm_residual = max(norm_residual)
        # if convergence criteria is met, end, else, update solution
        if norm_residual < residual_convergence_criteria:
            # converged solution found
            if verbose:
                print('\n\n\t\tharmonic balance solution found.\n')
            else:
                print('done.\n')
            break
        elif np.isnan(norm_residual) or np.isinf(norm_residual):
            # unstable solution
            if verbose:
                print('\n\n\t\tunstable solution. try again.\n')
            else:
                print('unstable solution. try again.\n')
            break
        else:
            
            
            # adjust the pseudo-timestep
            if adjust_delta_tau == True and iteration > 1:
                if spring_mass:
                    #current_res = max(residual_history[0][iteration],residual_history[1][iteration])
                    #previous_res = max(residual_history[0][iteration-2],residual_history[1][iteration-2])
                    current_res_1 = residual_history[0][iteration]
                    previous_res_1 = residual_history[0][iteration-2]
                    ratio_1 = previous_res_1/current_res_1
                    current_res_2 = residual_history[1][iteration]
                    previous_res_2 = residual_history[1][iteration-2]
                    ratio_2 = previous_res_2/current_res_2
                    ratio = min(ratio_1, ratio_2)
                else:
                    current_res = residual_history[iteration]
                    previous_res = residual_history[iteration-2]
                    ratio = previous_res/current_res         
                scaling_fac = 1.00001
                delta_tau_new = delta_tau*scaling_fac*ratio
                #delta_tau = delta_tau_new
                
                # switch to initial value
                if ratio < 1.0:
                    # the current residual is larger than the previous one
                    
                    '''
                    # HACK!!!
                    if iteration > 1000:
                        print('\nf_HB =', f_HB)
                        break
                    '''
                    
                    
                    # rewind the solution halfway and reset the timestep
                    if iteration > 1000:
                        f_HB = f_HB_history[iteration-1000]
                        delta_tau = delta_tau_init
                        

                    
                    '''
                    if going_up_count == 0:
                        unstable_iter = iteration
                    going_up_count += 1
                    #print('\n\nRESIDUAL GOING UP!')
                    unstable_iter -= 1
                    if unstable_iter > 1 and going_up_count <=2:
                        delta_tau = delta_tau_hist[unstable_iter-1]
                    else:
                        if delta_tau/2 > delta_tau_init:
                            delta_tau = delta_tau/2
                        else:
                            delta_tau = delta_tau_init
                    '''
                    
                else:
                    
                    going_up_count = 0
                    delta_tau = delta_tau_new
                print('\tdelta_tau =', delta_tau)
            # record time step used (keep it outside the if statment so that it
            # gets recorded at every iteration)
            delta_tau_hist.append(delta_tau)
            
            
            # update the solution
            if spring_mass:
                for m in range(n_masses):
                    f_HB[m] += error[m]
            else:
                f_HB += error
            # append the updated solution to the solution history
            f_HB_history.append(np.copy(f_HB))
                
        # if we've reached the maximum allowable number of pseudo-time steps
        if iteration == max_pseudo_steps-1:
            print('\n\t\tmaximum number of pseudo-time steps reached.\n')
        
    # if animation desired but no plot, make plot anyway and print warning
    if make_movie and not make_plot:
        print('\n\tcan\'t make movie without making plot...making plot.')
        make_plot = True
    # print message to the screen
    print('\nmaking a movie...quiet on the set!\n')
    # if desired, plot the results
    if make_plot:
        # plotting: USER INPUTS! want to animate the solution history or just
        # plot the final result? (True=animate, False=just print final result)
        animate_plot = make_movie
        plot_name = 'harmonic-balance 2 springs'
        n_images = iteration+1                  # total number of images computed
        skip_images = 90000                 # images to skip between frames
        auto_play = auto_play_movie     # automatically play the movie
        auto_open = auto_open_plot      # automatically open the final image
        # plotting: instantiate the figure
        fig = plt.figure(plot_name)
        # plotting: rescale the figure window to fit both subplots
        xdim, ydim = plt.gcf().get_size_inches()
        # for two plots, this scaling can't be more than 1.7!!!
        plt.gcf().set_size_inches(1.7*xdim, ydim, forward=True)
        # set the title for the HB solution plot
        title = ''
        counter=1
        for omega in omegas:
            title = title + '$\omega_{'+str(counter)+'} ='+str(omega)+'\quad $'
            counter += 1
        # extract the individual solutions for cases with multiple masses
        if spring_mass:
            '''
            # create separate lists of histories
            f_HB_histories = [[] for mass in range(n_masses)]
            for solution in f_HB_history:
                current_solutions = [[] for mass in range(n_masses)]
                for i in range(N):
                    for j in range(n_masses):
                        current_solutions[j].append(solution[i][j])
                for j in range(n_masses):
                    f_HB_histories[j].append(current_solutions[j])
            '''
            # set a color vector for the solutions
            colors = ['b','r']
        # things that won't change for the residual history plot
        plt.subplot(1,2,2)
        plt.xlabel('$iteration$', fontsize=16)
        plt.ylabel('$\\left\Vert \\frac{\partial f}{\partial t} \minus D_{HB}f_{HB} \\right\Vert_2$', fontsize=16)
        plt.title(r'$\Delta\tau = '+str(delta_tau)+'$')
        plt.xlim(0,iteration)
        if spring_mass:
            min_power = int(math.log(min(residual_history[0]),10))-1
            max_power = int(math.log(max(residual_history[1]),10))+1
        else:
            min_power = int(math.log(min(residual_history),10))-1
            max_power = int(math.log(max(residual_history),10))+1
        plt.ylim(pow(10,min_power), pow(10,max_power))
        # plotting: set the total number of frames
        if animate_plot == True:
            # capture all frames (skipping, if necessary) and the final frame
            all_frames = list(range(0,n_images,skip_images+1))+[n_images-1]
        else:
            # no animation: just capture the last one
            all_frames = [n_images-1]
        # plotting: capturing the movie
        writer = animation.writers['ffmpeg'](fps=10)
        with writer.saving(fig, plot_name+'.mp4', 300):
            frame = 0
            for n in all_frames:
                # plot the HB solution
                plt.subplot(1,2,1)
                plt.cla()
                if spring_mass:
                    for m in range(n_masses):
                        plt.plot(t_HB,f_HB_history[n][m],colors[m]+'o')
                        t_HB_int, f_HB_int, dummy = fourierInterp_given_freqs(t_HB, f_HB_history[n][m], omegas)
                        plt.plot(t_HB_int, f_HB_int, colors[m]+'--')
                else:
                    plt.plot(t_HB,f_HB_history[n],'mo')
                    t_HB_int, f_HB_int, dummy = fourierInterp_given_freqs(t_HB, f_HB_history[n], omegas)
                    plt.plot(t_HB_int, f_HB_int, 'm--')
                plt.xlabel('$t$', fontsize=16)
                plt.ylabel('$f_{HB}$', fontsize=16)
                plt.ylim(np.min(f_HB_history), np.max(f_HB_history))
                #plt.title(title)
                # plot the residual
                plt.subplot(1,2,2)
                if spring_mass:
                    for m in range(n_masses):
                        if n > 0 and residual_history[m][n] >= residual_history[m][0]:
                            plt.semilogy(residual_history[m][:n+1],'g-')
                            plt.semilogy(residual_history[m][:n+1], colors[m]+'--')
                        else:
                            plt.semilogy(residual_history[m][:n+1],'m-')
                            plt.semilogy(residual_history[m][:n+1], colors[m]+'--')
                else:
                    if n > 0 and residual_history[n] >= residual_history[0]:
                        plt.semilogy(residual_history[:n+1],'g-')
                    else:
                        plt.semilogy(residual_history[:n+1],'r-')
                if adjust_delta_tau:
                    plt.title(r'$\Delta\tau = '+str(delta_tau_hist[n])+'$')
                # set the spacing options
                #plt.tight_layout()
                #plt.subplots_adjust(left=0.07)
                #plt.subplots_adjust(right=0.95)
                # progress monitor
                percent_done = float(n)*100.0/(n_images-1)
                print('\tcapturing fig. '+plot_name+' (frame #'+str(frame) + \
                       '): '+str(round(percent_done,2))+'%')
                writer.grab_frame()
                frame += 1
            writer.grab_frame()
        # rescale the y-axis of the HB solution plot before saving an image
        plt.subplot(1,2,1)
        if spring_mass:
            white_space = (max(f_HB_history[-1].flatten())-min(f_HB_history[-1].flatten()))/5.0
            plt.ylim(min(f_HB_history[-1].flatten())-white_space,max(f_HB_history[-1].flatten())+white_space)
        else:
            white_space = (max(f_HB_int)-min(f_HB_int))/5.0
            plt.ylim(min(f_HB_int)-white_space,max(f_HB_int)+white_space)
        # plotting: save an image of the final frame
        print('\n\t'+'saving final image...', end='')
        file_name = plot_name+'.png'
        plt.savefig(file_name, dpi=500)
        print('figure saved: '+plot_name+'\n')
        # free memory used for the plot
        plt.close(fig)
        # start playing the movie, if desired
        if animate_plot and auto_play:
            webbrowser.open(plot_name+'.mp4')
        # open the saved image, if desired
        if auto_open:
            webbrowser.open(file_name)
        
    # return the full, converged solution (t_HB, f_HB)
    return t_HB, f_HB_history[-1]
#-----------------------------------------------------------------------------#

# instantiate a spring-mass problem object
the_ode_system = two_spring_mass_ode_system(my_two_spring_problem, initial_conditions)

# select a way to do the time-marching ('march together' or 'march separately')
ta_approach = 'march together'

# print a message to the screen 
print('\ncomputing time-accurate solution...\n')

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
f_0 = np.array([[f_1_0],
                [f_2_0]])
                
# explicit time marching, Euler's method
for n in range(1,n_time_points):
    
    if ta_approach == 'march together':    
        # update and append velocity vector
        if t[n-1]==t_0:
            v_n = v[n-1] + del_t*(the_ode_system.evaluate(x[n-1])+f_0)
        else:
            v_n = v[n-1] + del_t*the_ode_system.evaluate(x[n-1])
        v.append(v_n)
        # append the individual positions
        v_1.append(v_n[0][0])
        v_2.append(v_n[1][0])
    
    
    
    # recast the two carts' positions and the current time as 1x1 numpy arrays
    t_current = np.array(t[n-1]).reshape(1,1)    
    x_1_current = np.array(x_1[n-1]).reshape(1,1)
    x_2_current = np.array(x_2[n-1]).reshape(1,1)
    
    # print progress to the screen
    if n%100000==0:
        print('\titer: '+str(n)+'\tt = '+str(round(float(t_current),3)) + \
              ' s\tx_1 = '+str(round(float(x_1_current),3))+' m\tx_2 = ' + \
              str(round(float(x_2_current),3))+' m')
    
    if ta_approach == 'march separately':
        # make a list containing the current solution
        current_solution = [x_1_current, x_2_current]
        # update the velocity values
        v_1_next = v_1[n-1] + del_t*the_ode_system.evaluate_RHS_i(1, current_solution, t_current, static_initial_condition=True)[0]
        v_2_next = v_2[n-1] + del_t*the_ode_system.evaluate_RHS_i(2, current_solution, t_current, static_initial_condition=True)[0]
        # record these values
        v_1.append(v_1_next)
        v_2.append(v_2_next)
        # create velocity vector holding the updates values
        v.append(np.array([[v_1[n]],v_2[n]]))
    
    
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
#make_plots(t, x_1, x_2, x_1_0, x_2_0, l_1, l_2, wagon_1_width, wagon_2_width, auto_open=True)

# animate the time-accurate solution
#animate_wagon(t, x_1, x_2, v_1, v_2, wagon_1_width, wagon_2_width, auto_play=True)


'''

# take the DFT of both time traces
s, F, powers, x_1_peaks_found, x_1_peak_boundaries = my_dft(t, x_1, 
                                                         percent_energy_AC_peaks=98,
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
                                                         percent_energy_AC_peaks=99,
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

'''

# solve the system using HB
omegas = [2.736, 7.163]
x_0 = np.array([[initial_conditions['x_1_0']],
                [initial_conditions['x_2_0']]])

t_HB, f_HB = solve_HB_problem(omegas, 
                              time_discretization='use_T1', 
                              the_ode=the_ode_system, 
                              delta_tau=1e-8,
                              constant_init_guess=x_0, 
                              residual_convergence_criteria=5e-1, 
                              make_plot=True, 
                              auto_open_plot=True, 
                              make_movie=True, 
                              auto_play_movie=True, 
                              verbose=True, 
                              spring_mass=True)



print('\neigenvalues of the homogeneous system:')
print('\n    lambda_1:\t', lambda_1)
print('    lambda_2:\t', lambda_2)
print('    lambda_3:\t', lambda_3)
print('    lambda_4:\t', lambda_4)
print('\n    N.B. (1) the four eigenvalues are two complex-conjugate pairs.')
print('         (2) the magnitudes of these eigenvalues represent the \n' + \
      '             angular frequencies at which the wagons oscillate\n')

# see if the the time-accurate solutions corresponding to the HB time instances
# actually return a zero residual in the HB system
from HB_practice import harmonic_balance_operator
from time_spectral import linearInterp

D_HB, t_HB = harmonic_balance_operator(omegas, time_discretization='use_T1')

# extract the values of the time-accurate solutions at the last N-1 time 
# instances
t_HB,x_1_ta_at_t_HB = linearInterp(t, x_1, t_HB, verbose=True)
t_HB,x_2_ta_at_t_HB = linearInterp(t, x_2, t_HB, verbose=True)


plt.figure()
plt.plot(t_HB, x_1_ta_at_t_HB, 'bo')
plt.plot(t_HB, x_2_ta_at_t_HB, 'ro')
plt.plot(t,x_1,'b')
plt.plot(t,x_2,'r')
plt.plot(t_HB, f_HB[0],'b*--')
plt.plot(t_HB, f_HB[1],'r*--')
plt.xlim(t_HB[0],t_HB[-1]+t_HB[1])
plt.xlabel('$t$')
plt.ylabel('$x$')
f_minus=f_HB[0][0] - (1/2)*(max(f_HB[0])-min(f_HB[0]))*(1.0-np.cos(omegas[0]*t_HB))
f_plus=f_HB[0][0] + (1/2)*(max(f_HB[0])-min(f_HB[0]))*(1.0-np.cos(omegas[0]*t_HB))
plt.plot(t_HB,f_minus,'c.--',t_HB,f_plus,'c.--')
f_minus=f_HB[1][0] - (1/2)*(max(f_HB[1])-min(f_HB[1]))*(1.0-np.cos(omegas[0]*t_HB))
f_plus=f_HB[1][0] + (1/2)*(max(f_HB[1])-min(f_HB[1]))*(1.0-np.cos(omegas[0]*t_HB))
plt.plot(t_HB,f_minus,'m.--',t_HB,f_plus,'m.--')
plt.show()

# second derivative operator
D_HB2 = np.dot(D_HB,D_HB)


N = len(t_HB)
n_masses=2

x_ta_at_t_HB = [np.array(x_1_ta_at_t_HB).reshape(N,1), np.array(x_2_ta_at_t_HB).reshape(N,1)]

residual = []
norm_residual = []
for m in range(n_masses):
    func_evaluations = the_ode_system.evaluate_RHS_i(m+1, x_ta_at_t_HB, t_HB)
    matrix_vector_product = np.dot(D_HB2,x_ta_at_t_HB[m])
    residual.append(func_evaluations - matrix_vector_product)
    norm_residual.append(np.linalg.norm(residual[m]))
    print('\nfunc_evaluations_'+str(m+1)+' = ', func_evaluations)
    print('\nmatrix_vector_product_'+str(m+1)+' = ', matrix_vector_product)
    print('\nresidual_'+str(m+1)+' = ', residual[m])
    print('\nnorm_residual_'+str(m+1)+' = ', norm_residual[m])









