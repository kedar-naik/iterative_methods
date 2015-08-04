# -*- coding: utf-8 -*-
"""
Created on Fri Feb 06 14:12:31 2015

@author: Kedar
"""

# creating the time-spectral operator matrix
def time_spectral_operator(N,T):
    """
    This function returns the time-spectral operator matrix given a number of
    time instances and a period of oscillation.
    
    Inputs: 
      - N: number of time instances
      - T: period of oscillaton
    
    Output:
      - D: time-spectral operator matrix (as a list of list)
    """
    import math
    
    # initialize D
    D = [[float('nan') for i in range(N)] for j in range(N)]
    
    # build the time-spectral matrix
    for i in range(N):
        for j in range(N):
            
            if N%2 == 0:
                
                # for an even number of time instances
                if i == j:
                    D[i][j] = 0.0
                else:
                    D[i][j] = (math.pi/T)*pow(-1.0,(i-j))* \
                              (1/math.tan(math.pi*(i-j)/N))
            else:
                
                # for an odd number of time instances
                if i == j:
                    D[i][j] = 0.0
                else:
                    D[i][j] = (math.pi/T)*pow(-1.0,(i-j))* \
                              (1/math.sin(math.pi*(i-j)/N))
    return D


# this function returns the value of a user-defined function ##################
def myPeriodicSignal(t,T):
    """
    Samples the function defined below at the given time instances
  
    Input:
      - vector of time samples, t
      - period of oscillation, T
    Output:
      - discrete signal samples, f
      - discrete signal derivative, df/dt
      - discrete signal second derivative, df2dt2
    """
    import math
  
    # Type in your analytical function and dervatives here:
    #f = [pow(math.sin(2*math.pi*t_i/T),3)+pow(math.cos(2*math.pi*t_i/T),3) + 9 \
     #    for t_i in t]
    #dfdt = [(-1/T)*(3*math.pi*math.sin(4*math.pi*t_i/T)* \
      #      (math.cos(2*math.pi*t_i/T) - math.sin(2*math.pi*t/T))) for t_i in t] 
  
    f = [math.sin(2*math.pi*t_i/T)+9 for t_i in t]
    dfdt = [(2*math.pi/T)*math.cos(2*math.pi*t_i/T) for t_i in t]
    
    f = [math.cos(2*math.pi*t_i/T) for t_i in t]
    dfdt = [(-2*math.pi/T)*math.sin(2*math.pi*t_i/T) for t_i in t]
    df2dt2 = [-pow(2*math.pi/T,2)*math.cos(2*math.pi*t_i/T) for t_i in t]
    
    return (f, dfdt, df2dt2)
  
  
# this function returns the value of a user-defined 1st-order ODE ############
def myPeriodicODE(t,T,u):
    """
    Samples the RHS of an ODE of the form du/dt = f(u,t), where f(u,t) is 
    periodic with period T.
    
    Input:
      - time, t
      - period of oscillation, T
      - current solution, u
    Output:
      - evaluation of the RHS of the ODE, f(u)
    """
    import math
    
    # The equation for periodic population harvesting
    k = 0.5                   # growth rate of the population
    N = 10                    # carrying capacity
    h = 0.5                   # determines total rate of periodic harvesting
    b = 2*math.pi/T           # b = 2*pi/period of the sinusoidal function
    
    dudt = k*u*(1-(u/N)) - h*(1+math.sin(b*t))
    dudt = k*u*(1-(u/N)) - h*(1+math.sin(b*t)*math.cos(b*t)**4)
    
    return dudt
  
# this function returns a matrix-vector product, b, i.e. Ax ###################
def myMult(A,x):
    """
    Multiples m-by-n matrix A by n-by-1 vector x and returns m-by-1 vector b.
    
    Input:
      - matrix, A (as a list of lists)
      - vector, x (as a list)
    Output:
      - vector, b (as a list)
    """
  
    m = len(A)            # number of rows
    n = len(A[1])         # number of columns
    b = []
    for i in range(m):
        row_sum = 0
        for j in range(n):
            row_sum = row_sum + A[i][j]*x[j]
        b.append(row_sum)
    
    return b

# this function is linsapce for lists #########################################
def myLinspace(start,end,n):
    """
    This function returns a list of equally spaced points between the starting
    value and the ending value. The desired number of entries in the list is 
    provided by the user.
    
    Input:
      - first value, start
      - final value, end
      - desired number of total points, n
    Output:
      - list of n points, points 
    """
    
    points = []
    
    try:
        points = [start+index*float(end-start)/(n-1) for index in range(n)]
        
    except ZeroDivisionError:
        print "\n Oh no! \n The number of desired points given to " + \
        "myLinsapce() must be greater than or equal to two! \n " + \
        "Otherwise, there is a divide-by-zero error!"
    
    if len(points) == 0:
        print "\n Please check the inputs to the myLinspace function." + \
        "\n The list of points is empty! Something has gone wrong..."
    else:
        return points
    
# this function interpolates a series of points with a Fourier series #########
def fourierInterp(x,y):
    
    """
    This function interpolates a given set of ordinates and abscissas with a
    Fourier series. The function returns a Fourier interpolation (of the 
    highest degree trig polynomial allowed by the Nyquist Criterion) along with 
    the corresponding new set of abscissas, which are ten times as finely 
    spaced as the original. The first derivative of the interpolant is also 
    returned. Note that the interpolants will only be exact if the given points
    are just one shy of representing an exact period
    
    Input:
      - abscissas, x (as a list) (leave out last, duplicate point in period)
      - ordinates, y (as a list) (again, leave out last point, if periodic)
    Output:
      - new abscissas, x_int (as a list)
      - interpolated ordinates, y_int (as a list)
      - derivative of the interpolant, dydx_int (as a list)
    """
    
    import math
    
    # refinment factor for the interpolant. (If there are originally 10 points
    # but you want the Fourier Series to be defined on 50 points, then the 
    # refinement factor is 5.)
    refine_fac = 10
    
    # preliminaries
    n = len(x)                  # number of abscissas
    n = float(n)                # for mathematical clarity in the next line
    m = math.floor((n-1)/2)     # highest degree trig polynomial allowed by NC
    
    # establish scaling to the domain [0,2pi)
    x_interval = x[1]-x[0]      # even interval in abscissas
    period = n*x_interval       # extrapolate and find length of the period
    
    # initalization
    n = int(n)          # recast as int
    m = int(m)          # recast as int
    a = [0.0]*(m+1)       # "+1" in order to incorporate "DC value" at a_0
    b = [0.0]*(m+1)       # b_0 never used, but included to match length of a
    
    # compute the coefficients
    for j in range(m+1):
        a[j] = 0.0
        b[j] = 0.0
        for i in range(n):
            scaled_x = (2*math.pi/period)*(x[i]-x[0])
            a[j] += (2.0/n)*math.cos(j*scaled_x)*y[i]
            b[j] += (2.0/n)*math.sin(j*scaled_x)*y[i]
    
    # find the actual interpolation
    n_int = refine_fac*(n+1)
    x_int = myLinspace(x[0],x[-1]+x_interval,n_int)
    y_int = [0.0]*n_int
    dydx_int = [0.0]*n_int
    for i in range(n_int):
        y_int[i] = a[0]/2.0    # the "DC" value
        dydx_int[i] = 0.0      # intialize the summation
        scaled_x_int = (2*math.pi/period)*(x_int[i]-x[0])
        for j in range(m):
            y_int[i] += a[j+1]*math.cos((j+1)*scaled_x_int) + \
                        b[j+1]*math.sin((j+1)*scaled_x_int)
            dydx_int[i] += (2*math.pi/period)*(j+1)* \
                           (b[j+1]*math.cos((j+1)*scaled_x_int) - \
                            a[j+1]*math.sin((j+1)*scaled_x_int))
    
    return (x_int, y_int, dydx_int)
    
# this function extracts the period of steady-state oscillations ##############
def extractPeriod(t,f):
    """
    Given a time-accurate solution, this function will return the period and
    the interpolated time history over one period. The following algorithm will
    be used:
    -interpolate the last third of the given time history with a Fourier series
    -find the maximum function value from the end of the interpolant
    -find a handful of points that are closest to the maximum
    -group those points into different peak clusters
    -find the average time corresponding to each cluster
    -define period as average time interval between average peak times
    
    Input:
      - time samples, t (as a list)
      - function values, f (as a list)
    Output:
      - period, T
      - time samples over a period, t_period (as a list)
      - function values over a period, f_period (as a list)
    """
    from matplotlib import pylab as plt
    import math
    
    # length of the time history
    n = len(t)
    
    # assume that steady-state behavior appears ~2/3 of the way in
    # N.B. n and 3 are ints, so floor() or ceil() are not needed.
    t_end = t[-n/3:]
    f_end = f[-n/3:]
    
    # interpolate these points using a Fouier series
    t_end_int, f_end_int, dummy = fourierInterp(t_end, f_end)
    
    # since the above Fourier interpolation function assumes perfect 
    # periodicity of the points given to it, the fit will likely be quite
    # inaccurate at the ends. So, only use the middle 70% of the interpolation
    # by cutting off the first and last 15%.
    cut_percent = 15
    frac = cut_percent/100.0
    n = len(t_end_int)
    t_clean = t_end_int[int(math.ceil(n*frac)):int(math.floor(-n*frac))]
    f_clean = f_end_int[int(math.ceil(n*frac)):int(math.floor(-n*frac))]
       
    # find max and min function values present in the last third of the 
    # "cleaned" data. this way, if any transient effects have made their way 
    # into the data, they hopefully won't corrupt the result
    n = len(f_clean)    
    max_f = max(f_clean[-n/3:])
    min_f = min(f_clean[-n/3:])
    
    # find a handful of other points that come closest to the maximum. start 
    # with a small tolerance and then keep doubling it until the desired number
    # of other points has been found.
    other_points = 16
    tol = 1e-8
    indices_maxes = []
    while len(indices_maxes) < other_points+1:
        indices_maxes = [i for i in range(n) if f_clean[i] > max_f-tol]
        tol *= 2.0
        
    # now find the other points closest to the minimum
    tol = 1e-8
    indices_mins = []
    while len(indices_mins) < other_points+1:
        indices_mins = [i for i in range(n) if f_clean[i] < min_f+tol]
        tol *= 2.0
    
    # plotting: these lists are for plotting purposes only
    t_maxes = [t_clean[i] for i in indices_maxes]
    f_maxes = [f_clean[i] for i in indices_maxes]
    t_mins = [t_clean[i] for i in indices_mins]
    f_mins = [f_clean[i] for i in indices_mins]
    
    # plotting: instantiate the figure and plot some results
    plt.figure()
    plt.plot(t_end,f_end,'ko',label='$last\,1/3$')
    plt.plot(t_end_int,f_end_int,'k--',label='$Fourier\,interp.$')
    plt.plot(t_clean,f_clean,'r-',label='$trimmed$')
    plt.plot(t_maxes, f_maxes, 'y.', label='$near\,tips$')
    plt.plot(t_mins, f_mins, 'y.')
    
    #-------------------------------------------------------------------------#
    def points_to_period(indices, t, f):
        """
        This function takes the list of extremum indices (either indices_max or
        indices_min) and clusters them into groups based on their spacing. It 
        then takes each cluster and finds an average corresponding time value. 
        
        Input:
          - list of indices corresponding to extrema, indices
          - larger list of time points over which extrema were found, t
          - larger list of function values, f (N.B. only needed for plotting)
        Output:
          - period based on the given set of extrema, T_extremum
        """
        # go through the list of maxima or minima and separate into clusters 
        # any points that are only one index apart. cluster is a list of lists.
        cluster = []
        cluster.append([indices[0]])
        n_clusters = 1
        for i in range(len(indices)-1):
            if indices[i+1] == indices[i]+1:
                cluster[n_clusters-1].append(indices[i+1])
            else:
                cluster.append([indices[i+1]])
                n_clusters += 1
    
        # average the time values corresponding to each cluster
        t_tips = []
        f_tips = []                  # needed only for plotting
        for i in range(n_clusters):
            cluster_times = [t[entry] for entry in cluster[i]]
            t_tips.append(sum(cluster_times)/len(cluster[i]))
            # needed only for plotting
            cluster_func_vals = [f_clean[entry] for entry in cluster[i]]
            f_tips.append(sum(cluster_func_vals)/len(cluster[i]))

        # plot the average maxes or mins
        plt.plot(t_tips,f_tips,'b*',label='$max/min \, tips$')
        
        # for the tips found, find the time intervals in between adjacent ones
        t_intervals = []    
        for i in range(len(t_tips)-1):
            t_intervals.append(t_tips[i+1]-t_tips[i])
    
        # define the period as being the average time interval found above
        T_extremum = sum(t_intervals)/(len(t_tips)-1)
        
        return T_extremum
    #-------------------------------------------------------------------------#
    
    # use the lists of maxes and mins and get the corresponding periods
    T_peaks = points_to_period(indices_maxes, t_clean, f_clean)
    T_troughs = points_to_period(indices_mins, t_clean, f_clean)
    
    # define period as the average value of the two values for period above
    T = (T_peaks + T_troughs)/2
    
    # round period to two decimal points 
    T = round(T,2)
    
    # print results to the screen
    print '\nperiod-extraction results:'
    print '  T_peaks = ', T_peaks
    print '  T_troughs = ', T_troughs
    print '  T = ', T
    
    # plotting: finish plot of the period-extraction process
    plt.rc('text', usetex=True)               # for using latex
    plt.rc('font', family='serif')            # setting font
    plt.xlabel(r'$t$', fontsize=18)
    plt.ylabel(r'$f(t)$', fontsize=18)
    plt.legend(loc='best', ncol=1)
    plt.title(r'$T = \,$'+str(T))
    # plotting: save image
    plot_name = 'period extraction'
    print 'saving image...'
    plt.savefig(plot_name, dpi=1000)
    print 'figure saved: ' + plot_name
    #plotting: free memory
    plt.close()
    
    # recover one peak-to-peak time trace of the interpolation
    delta_t = t_clean[1]-t_clean[0]
    points_period = int(round(T/delta_t))
    last_period = f_clean[-points_period:]
    max_f = max(last_period)
    max_index = last_period.index(max_f)
    max_index_clean = -points_period+max_index
    t_period = myLinspace(0,T,points_period+1)
    f_period = f_clean[max_index_clean-points_period:max_index_clean+1]
    
    return (T, t_period, f_period)
    
# this functions returns the l2-norm of a given list ##########################
def myNorm(x):
    """
    This function returns the l2-norm of a given list of numbers.
    
    Input:
      - vector, x (as a list) 
    Output:
      - l2-norm, norm
    """
    
    import math
    
    abs_sq = [abs(float(entry))**2 for entry in x]
    norm = math.sqrt(sum(abs_sq))
    
    return norm
    
# main ########################################################################
def main():
    import math
    import matplotlib                        # import by itself first
    matplotlib.use('Agg')                    # use Anti-Grain Geometry backend
    from matplotlib import pylab as plt      # must be called AFTER use()
    from matplotlib import animation         # for specifying the writer

    # close open windows
    plt.close('all')
        
    # user inputs
    N = 17                  # number of time instaces
    T = 2*math.pi          # period of osciallation (enter a float!)
    T=2.0
    
    # create the time-spectral operator 
    D = time_spectral_operator(N,T)
    
    # time interval
    delta_T = T/N
         
    # indices corresponding to the time instances
    indices = range(N)
         
    # list of time instances
    t = [delta_T*index for index in indices]
    
    ###########################################################################
    # Check to see if the time-spectral operator is working ###################
    ###########################################################################
    
    # sampling at the time instances
    f_TS, dummy1, dummy2 = myPeriodicSignal(t,T)
    
    # the time derivatives at the time instances using time-spectral operator
    dfdt_TS = myMult(D, f_TS)
    
    # find the second derviative using the time-spectral operator as well
    df2dt2_TS = myMult(D, dfdt_TS)
    
    # interpolate the time-spectral results with Fourier series
    t_int, dfdt_TS_int, dummy1 = fourierInterp(t, dfdt_TS)
    t_int, df2dt2_TS_int, dummy1 = fourierInterp(t, df2dt2_TS)
    
    # fine time grid (10 times the number of time instances) for "exact" values
    t_fine = [T*index/(10*N-1) for index in range(10*N)]
    f_fine, dfdt_fine, df2dt2_fine = myPeriodicSignal(t_fine,T)
    
    # plotting: USER INPUT. Would you like to plot this verification figure?\
    plot_figure = False
    plot_name = 'TS verification'
    
    if plot_figure == True:
        # plotting
        print '\nplotting fig. ' + plot_name + '...'
        plt.rc('text', usetex=True)               # for using latex
        plt.rc('font', family='serif')            # setting font
        # plot the "exact" results
        plt.plot(t_fine,f_fine,'k-',label='$f$')
        plt.plot(t_fine,dfdt_fine,'r-',label='$df/dt$')
        plt.plot(t_fine,df2dt2_fine,'b-',label='$d^2f/dt^2$')
        # plot the time instances
        plt.plot(t,f_TS,'ko',label='$f_{TS}$')
        # plot the time-spectral first dervative and interpolation
        plt.plot(t,dfdt_TS,'go',label='$df_{TS}/dt')
        plt.plot(t_int,dfdt_TS_int,'g--',label='$(Fourier \, Interp.)$')
        # plot the time-spectral second dervative and interpolation
        plt.plot(t,df2dt2_TS,'yo',label='$d^2f_{TS}/dt^2')
        plt.plot(t_int,df2dt2_TS_int,'y--',label='$(Fourier \, Interp.)$')
        # limits, labels, legend, title
        plt.xlabel(r'$t$', fontsize=18)
        plt.ylabel('')
        plt.legend(loc='best', ncol=2)
        plt.title(r'$N = \,$'+str(N))
        # save figure
        print 'saving fig. ' + plot_name + '...'
        plt.savefig(plot_name, dpi=1000)
        print 'fig.' + plot_name + ' saved'
        plt.close()
    
    ###########################################################################
    # [time accurate] explicit euler ##########################################
    ###########################################################################
    delta_t = 0.05            # time step
    initial_value = 8        # intitial condition
    t_end = 25               # final time
  
    f = []
    times = []
    time_points = int((t_end/delta_t)+1)
    
    # time stepping
    for n in range(time_points):
        times.append(n*delta_t)
        if n == 0:
            # record the initial condition
            f.append(initial_value)
        else:
            # explicitly step forward in time 
            f.append(f[n-1] + delta_t*myPeriodicODE(times[n-1],T,f[n-1]))
            
    # plotting: USER INPUTS! do you want to animate the solution history or just
    # plot the final result? (True = animate, False = just print final result)
    animate_plot = False                     
    plot_name = 'time-accurate ODE'
    n_images = time_points            # total number of images computed
    skip_images = 1                   # images to skip between animation frames
    
    # plotting: initializations
    fig = plt.figure()
    l, = plt.plot([], [],'k-',label='f')

    # plotting: things that will not be changing inside the loop
    plt.rc('text', usetex=True)               # for using latex
    plt.rc('font', family='serif')            # setting font
    plt.xlabel(r'$t$', fontsize=18)
    plt.ylabel(r'$f(t)$', fontsize=18)
    plt.xlim(0,20)
    plt.ylim(7.8,9.0)
    plt.title(r'$\Delta t = \,$'+str(delta_t))
    
    # plotting: set the total number of frames
    if animate_plot == True:
        # capture all frames (skipping, if necessary) and the final frame
        all_frames = range(0,n_images,skip_images+1)+[n_images-1]
    else:
        # no animation: just capture the last one
        all_frames = [n_images-1]
    
    # plotting: capturing the movie
    writer = animation.writers['ffmpeg'](fps=15)
    with writer.saving(fig, plot_name+'.mp4', 100):
        frame = 0
        for n in all_frames:
            l.set_data(times[:n+1],f[:n+1])
            # progress monitor
            percent_done = float(n)*100.0/(n_images-1)
            print 'capturing fig. '+plot_name+' (frame #'+str(frame)+'): ', \
                   percent_done,'%'
            writer.grab_frame()
            frame += 1
        writer.grab_frame()
    
    # plotting: save an image of the final frame
    print 'saving final image...'
    plt.savefig(plot_name, dpi=1000)
    print 'figure saved: ' + plot_name
    
    # free memory used for the plot
    plt.close(fig)
    
    ###########################################################################
    # [time accurate] extract period from time-marching solution ##############
    ###########################################################################
    
    # pass the time-accurate solution history to the extractPeriod function
    T, t_period, f_period = extractPeriod(times,f)
    
    # plot an isolated period by itself
    plt.figure()
    plt.plot(t_period,f_period,'k-')
    plt.rc('text', usetex=True)               # for using latex
    plt.rc('font', family='serif')            # setting font
    matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{mathtools}"]
    plt.xlabel(r'$t$', fontsize=18)
    plt.ylabel(r'$f(t)$', fontsize=18)
    plt.title(r'$\textit{steady-state period extracted from '+\
                r'time-accurate result}$')
    plot_name = 'period extraction check'
    print 'saving image...'
    plt.savefig(plot_name, dpi=1000)
    print 'figure saved: ' + plot_name
    plt.close()
    
    ###########################################################################
    # [time spectral] explict pseudo-timestepping (dfdt -> f) #################
    ###########################################################################
    delta_tau = 0.0001            # pseudo-timestep
    init_value = 8.0             # constant intital guess
    max_pseudosteps = 500000      # maximum number of pseudo-timesteps to try
    conv_criteria = 1e-4         # resdiual convergence criteria
    
    # Initialize history lists
    res_hist = []                # residual history
    f_TS_hist =[]                # solution history
    
    # set the intial guess for the periodic solution
    f_TS = [init_value for index in indices]    # N.B. f_TS is being reassigned
    f_TS_hist.append(f_TS)   
    
    # pseudo-timestepping
    for iteration in range(max_pseudosteps):
                
        # compute D*f
        Df = myMult(D,f_TS)
        
        # compute dfdt from the ODE
        dfdt = [myPeriodicODE(t[index],T,f_TS[index]) for index in indices]
        
        # compute the residual vector for this solution
        res = [-Df[index]+dfdt[index] for index in indices]
        
        # find the norm of the residual vector and store it
        res_hist.append(myNorm(res))
        
        # print new residual to the screen
        print 'pseudo-time iteration #',iteration, \
              '; residual = ',res_hist[iteration]
        
        # check residual stopping condition
        if res_hist[iteration] < conv_criteria:
            print "\nsolution found.", iteration, " iterations required.\n" 
            break
        
        # update solution vector
        f_TS = [f_TS[index]+delta_tau*res[index] for index in indices]
                
        # store new solution for plotting
        f_TS_hist.append(f_TS)
    
    # plotting: user input! do you want to animate the solution history or just
    # plot the final result? (True = animate, False = just print final result)
    animate_plot = False                     
    plot_name = 'TS explicit pseudo'
    n_images = iteration+1
    skip_images = 5000
    
    # plotting: initializations and sizing
    fig = plt.figure()
    xdim, ydim = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches(2*xdim, ydim, forward=True)

    # plotting: things that will not be changing inside the loop
    plt.rc('text', usetex=True)               # for using latex
    plt.rc('font', family='serif')            # setting font
    # solution history plot
    plt.subplot(1,2,1)
    dots, = plt.plot([], [],'ko',label='$f_{TS}$')
    line, = plt.plot([], [],'k--',label='$ Fourier \, interp.$')
    plt.xlabel(r'$t$', fontsize=18)
    plt.ylabel(r'$f\left(t\right)$', fontsize=18)
    ave_init_value = sum(f_TS_hist[0])/N
    max_final = max(f_TS_hist[-1])
    min_final = min(f_TS_hist[-1])
    ampl_final = abs(max_final-min_final)
    white_space = ampl_final/3.0
    plt.xlim(0,T)
    if animate_plot == True:
        plt.ylim(ave_init_value, max_final+white_space)
    else:
        plt.ylim(min_final-white_space,max_final+white_space)
    # residual history plot
    plt.subplot(1,2,2)
    res, = plt.semilogy([], [],'b-',label='residual')
    plt.xlabel(r'$iteration$', fontsize=18)
    plt.ylabel(r'$\|R\|$', fontsize=18)
    plt.title(r'$\Delta\tau = '+str(delta_tau)+'$')
    plt.xlim(0,iteration)
    plt.ylim(1e-5,10)
    
    # plotting: set the total number of frames
    if animate_plot == True:
        # capture all frames (skipping, if necessary) and the final frame
        all_frames = range(0,n_images,skip_images+1)+[n_images-1]
    else:
        # no animation: just capture the last one
        all_frames = [n_images-1]
        
    # plotting: capturing the movie
    writer = animation.writers['ffmpeg'](fps=10)
    with writer.saving(fig, plot_name+'.mp4', 100):
        frame = 0
        for n in all_frames:
            # plot solution and interpolation
            plt.subplot(1,2,1)
            dots.set_data(t,f_TS_hist[n])
            t_int,f_TS_int, dummy1 = fourierInterp(t,f_TS_hist[n])
            line.set_data(t_int,f_TS_int)
            plt.title(r'$iteration \,\#$'+'$'+str(n)+'$')
            plt.legend(loc='best')
            # plot residual            
            plt.subplot(1,2,2)
            res.set_data(range(n),res_hist[:n])
            # progress monitor
            frame += 1
            percent_done = float(n)*100.0/(n_images-1)
            print 'capturing fig. '+plot_name+' (frame #'+str(frame)+'): ', \
                   percent_done,'%'
            writer.grab_frame()
        writer.grab_frame()
        
    # plotting: rescale the final frame to focus on the converged solution
    if animate_plot == True:
        plt.subplot(1,2,1)
        plt.ylim(min_final-white_space,max_final+white_space)
    
    # plotting: save an image of the final frame
    print 'saving final image...'
    plt.savefig(plot_name, dpi=1000)
    print 'figure saved: ' + plot_name
    
    # free memory used for the plot
    plt.close(fig)
    
    ###########################################################################
    # compare the time-spectral results against the time-accurate one #########
    ###########################################################################
    
    # plot both isolated period atop one another
    plt.figure()
    plt.plot(t_period,f_period,'b-', label=r'$\textit{time-accurate}$')
    plt.plot(t,f_TS_hist[-1],'ko')
    t_int,f_TS_int, dummy1 = fourierInterp(t,f_TS_hist[-1])
    plt.plot(t_int,f_TS_int,'k--', label=r'$\textit{time-spectral } \left(N='+str(N)+r'\right)$')
    plt.rc('text', usetex=True)               # for using latex
    plt.rc('font', family='serif')            # setting font
    matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{mathtools}"]
    plt.xlabel(r'$t$', fontsize=18)
    plt.ylabel(r'$f(t)$', fontsize=18)
    plt.legend(loc='best')
    plot_name = 'comparison - TS vs TA'
    print 'saving image...'
    plt.savefig(plot_name, dpi=1000)
    print 'figure saved: ' + plot_name
    plt.close()
    
# standard boilerplate to call the main() function
if __name__ == '__main__':
    main()