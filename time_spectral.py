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
    f = [math.sin(2*math.pi*t_i/T)+9 for t_i in t]
    dfdt = [(2*math.pi/T)*math.cos(2*math.pi*t_i/T) for t_i in t]
    
    f = [math.cos(2*math.pi*t_i/T) for t_i in t]
    dfdt = [(-2*math.pi/T)*math.sin(2*math.pi*t_i/T) for t_i in t]
    df2dt2 = [-pow(2*math.pi/T,2)*math.cos(2*math.pi*t_i/T) for t_i in t]
    
    # type out the expression you have entered as a raw string in latex form
    two_over_T = 2.0/T
    if two_over_T != 1.0:
        name = r'f(t)=cos('+str(two_over_T)+'\pi t)'
    else:
        name = r'f(t)=cos(\pi t)'
    
    return (f, dfdt, df2dt2, name)
  
# this function returns the value of a user-defined 1st-order ODE ############
def myPeriodicODE(t,u):
    """
    Samples the RHS of an ODE of the form du/dt = f(u,t), where f(u,t) is 
    periodic with period T.
    
    Input:
      - time, t
      - current solution, u
    Output:
      - evaluation of the RHS of the ODE, f(u)
    """
    import math
    
    # period of oscillation
    T = 2.0
    
    # The equation for periodic population harvesting
    k = 0.5                   # growth rate of population (Malthusian param.)
    C = 10                    # carrying capacity
    h = 0.5                   # determines total rate of periodic harvesting
    b = 2*math.pi/T           # b = 2*pi/period of the sinusoidal function
    
    # expression giving the derivative    
    dudt = k*u*(1-(u/C)) - h*(1+math.sin(b*t))
    dudt = k*u*(1-(u/C)) - h*(1+math.sin(b*t)*math.cos(b*t)**4.0)
    
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
        print("\n Oh no! \n The number of desired points given to " + \
        "myLinsapce() must be greater than or equal to two! \n " + \
        "Otherwise, there is a divide-by-zero error!")
    
    if len(points) == 0:
        print("\n Please check the inputs to the myLinspace function." + \
        "\n The list of points is empty! Something has gone wrong...")
    else:
        return points
    
# this function interpolates a series of points with a Fourier series #########
def fourierInterp(x, y, x_int=None):
    
    """
    This function interpolates a given set of ordinates and abscissas with a
    Fourier series. The function returns a Fourier interpolation (of the 
    highest degree trig polynomial allowed by the Nyquist Criterion) on the 
    given grid of new abscissas. If no vector of desired abscissas is given, 
    the set of interpolant abscissas is set automatically to be ten times as 
    finely spaced as the original. The first derivative of the interpolant is 
    also returned. Note that the interpolants will only be exact if the given 
    points are just one shy of representing an exact period.
    
    Input:
      - abscissas, x (as a list) (leave out last, duplicate point in period)
      - ordinates, y (as a list) (again, leave out last point, if periodic)
      - new abscissas, x_int (as a list) (optional! defaults to 10x refinement)
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
    
    # set x_int, if it hasn't been given
    if x_int == None:
        n_int = refine_fac*(n)
        x_int = myLinspace(x[0],x[-1]+x_interval,n_int)
    else:
        n_int = len(x_int)
    
    # find the actual interpolation
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

# this function linearly interpolates a series of points ######################
def linearInterp(x, y, x_int=None, verbose=False):
    
    """
    This function interpolates a given set of ordinates and abscissas with line
    segments between given points onto the given grid of new abscissas. If no 
    vector of desired abscissas is given, the set of interpolant abscissas is 
    set automatically to include as many new points between the original ones
    as specified below (set to 2, by default).
    
    Input:
      - abscissas, x (as a list) (leave out last, duplicate point in period)
      - ordinates, y (as a list) (again, leave out last point, if periodic)
      - new abscissas, x_int (as a list) (optional! defaults to 2x refinement)
      - print progress, verbose (bool) (optional! defaults to no output)
    Output:
      - new abscissas, x_int (as a list)
      - interpolated ordinates, y_int (as a list)
    """
    
    # setting the desired level of refinement of the interpolant abscissas.
    # enter the desired number of points you'd like to have appear beetween 
    # neighboring original abscissas. (e.g. if the original points are [0,2,4]
    # and you want the interpolant to be defined on [0,1,2,3,4], then the 
    # points_between is 1.)
    points_between = 2
    
    # number of original abscissas
    n = len(x)                  
    
    # if x_int hasn't been given, set it
    if x_int == None:
        x_int = []
        for i in range(n-1):
            x_int.append(x[i])
            interval = (x[i+1]-x[i])/float(points_between+1)
            for k in range(points_between):
                x_int.append(x_int[-1]+interval)
        x_int.append(x[-1])
    
    # number of interpolant abscissas
    n_int = len(x_int)
    
    # compute the interpolant
    y_int = []
    for i in range(n_int):
        # for each x_int, find the two closest x values
        distances = []
        for k in range(n):
            distances.append(abs(x_int[i]-x[k]))
        sorted(distances)
        indices = [distances.index(entry) for entry in sorted(distances)]
        first_two_indices = indices[:2]
        i1 = min(first_two_indices)
        i2 = max(first_two_indices)
        # if this x_int happens to be exactly halfway between two x values
        if i1 == i2:
            i2 = i1 + 1
        # find the corresponding x and y values
        x1 = x[i1]
        x2 = x[i2]
        y1 = y[i1]
        y2 = y[i2]
        # use the y values corresponding to the two x values to interpolate
        y_int.append(y1 + (y2-y1)*(x_int[i]-x1)/(x2-x1))
        
        # print progress to the screen, if requested
        if verbose:
            if round(i*100.0/n_int) % 10 == 0:
                print('linear interpolation: '+str(round(i*100.0/n_int,2))+'% done')
            
    return (x_int, y_int)
    
# this function extracts the period of steady-state oscillations ##############
def extractPeriod(t,f):
    """
    Given a time-accurate solution, this function will return the period and
    the interpolated time history over one period. The following algorithm will
    be used:
    -interpolate the last third of the given time history with a Fourier series
    -shave off the ends of the interpolant to account for inaccuracies
    -find the max and min function value from the end of the interpolant
    -find a handful of points that are closest to the max and min
    -group those points into different peak/trough clusters
    -find the average time corresponding to each cluster
    -define period as average time interval between average peak/trough times
    -define overall period as the average of periods found from peaks/troughs
    
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
    t_end = t[-int(n/3):]
    f_end = f[-int(n/3):]
    
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
    max_f = max(f_clean[-int(n/3):])
    min_f = min(f_clean[-int(n/3):])
    
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
        The average time span between clusters is taken to be the period for
        the given set of extrema.
        
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
    print('\nperiod-extraction results:')
    print('  T_peaks =', T_peaks)
    print('  T_troughs =', T_troughs)
    print('  T =', T)
    
    # plotting: finish plot of the period-extraction process
    #plt.rc('text', usetex=True)               # for using latex
    #plt.rc('font', family='serif')            # setting font
    plt.xlabel(r'$t$', fontsize=18)
    plt.ylabel(r'$f(t)$', fontsize=18)
    plt.legend(loc='best', ncol=1)
    plt.title(r'$T = \,$'+str(T))
    # plotting: save image
    plot_name = 'period extraction process'
    print('\nsaving image...')
    plt.savefig(plot_name, dpi=300)
    print('figure saved: ' + plot_name + '\n')
    #plotting: free memory
    plt.close()
    
    # recover one peak-to-peak time trace of the interpolation (this portion 
    # should always be valid and not need to be changed)
    delta_t = t_clean[1]-t_clean[0]
    points_period = int(round(T/delta_t)+1)
    last_period = f_clean[-points_period:]
    max_f = max(last_period)
    max_index = last_period.index(max_f)
    max_index_clean = -points_period+max_index
    start_at_clean = max_index_clean-points_period+1
    max_t = t_clean[max_index_clean]
    t_period_0 = t_clean[start_at_clean:max_index_clean+1]
    f_period_0 = f_clean[start_at_clean:max_index_clean+1]
    
    # N.B. We will assume that the point (t_max, f_max) is the actual peak of 
    # the steady-state period. (It might not be, of course: the discretization
    # might have just missed the actual peak.) We will take this peak and force 
    # it to represent the value of T in t_period, by subtracting off (t_max-T)
    # from the period_points times behind t_max. 
    t_period_0 = [entry-(max_t-T) for entry in t_period_0]
    
    # N.B. The first point above will most probably NOT correspond to zero!!! 
    # (It will be off by eps <= delta_t/2.) In order to get the first point to 
    # correspond to zero, without strechting or squuezing the actual curve, we 
    # will do a linear interpolation to a uniform grid within the range [0,T].
    # In order to speed up the final interpolation, the final time trace will 
    # be returned on a grid 1/10 as fine as that used for computing the period
    # above, i.e. points_period/10. (This may be changed by the user, if 
    # desired.)
    t_period = myLinspace(0,T,int(points_period/5))
    t_period, f_period = linearInterp(t_period_0, f_period_0, t_period, verbose=True)
    
    return (T, t_period, f_period)
    
# this function returns the p-norm (defaults to the l2-norm) of a vector ######
def myNorm(x, p=2):
    """
    This function returns the p-norm of a given vector. If no value for p is 
    provided, the l2-norm is a returned. The infinity norm is returned by 
    specifying p='inf'.
    e.g.
      - p=1:        l1-norm, "Taxicab norm"
      - p=2:        l2-norm, "Euclidian norm"
      - p='inf':    infinity norm
    
    Input:
      - vector, x (as a list) 
      - type of norm, p (defaults to p=2. p='inf' gives the infinity norm)
    Output:
      - l2-norm, norm
    """
    
    assert (p >= 1.0), 'p-norm not defined for p < 1'
        
    if p == 'inf':
        # for the infinity norm
        abs_x = [abs(entry) for entry in x]
        norm = max(abs_x)
    else:
        # for any other p-norm
        abs_p = [abs(float(entry))**p for entry in x]
        norm = sum(abs_p)**(1.0/p)
    
    return norm

# this function takes a TS solution and time-marches each instance up a period
def extend_instances(t, f_ts, T, make_plot=False):
    """
    Given a time-spectral solution, this function takes the value at each time
    instance and treats it as the intial condition for a time-accurate problem.
    This value is then explicitly advanced in time using Euler's method by a 
    duration of T. The values at the end of these time-marched segments are 
    compared to the original time instances. The function returns the l2-norm 
    of the differences.
    
    Input:
      - the time instances, t (as a list)
      - time-spectral solution, f_ts (as a list)
      - the assumed period, T
      - whether or not to plot the segments, make_plot (defaults to False)
      
    Output:
      - some function of the l2-norm of the errors found at the time instances
      - if make_plot is set to True, then an open, unsaved figure is created
    """
    import matplotlib                        # import by itself first
    matplotlib.use('Agg')                    # use Anti-Grain Geometry backend
    from matplotlib import pylab as plt      # must be called AFTER use()
    
    # number of time instaces
    N = len(t)
    
    # time-marching parameters
    points_per_T = 200
    
    # time step for time marching
    delta_t = T/(points_per_T-1)
    
    # initialize containers to hold the segments
    t_segments = []
    f_segments = []
    
    # step through the time instances
    for i in range(N):
        # set initial condition equal to the (t,f) of the this time instance
        t_start = t[i]              # initial condition time
        initial_value = f_ts[i]     # intitial condition value
        # intitalize lists for the individual segment
        f_segment = []
        t_segment = []
        # time stepping
        for n in range(points_per_T):
            t_segment.append(t_start+n*delta_t)
            if n == 0:
                # record the initial condition
                f_segment.append(initial_value)
            else:
                # explicitly step forward in time 
                f_segment.append(f_segment[n-1] + delta_t*myPeriodicODE(t_segment[n-1],f_segment[n-1]))
        # add the this short trace to the list of all segments
        t_segments.append(t_segment)
        f_segments.append(f_segment)
        
    # extract the points making up the extended period (end of each segment)
    f_extended = [segment[-1] for segment in f_segments]
    
    # compute the difference between the TS solution and the extended period
    errors = [(f_ts[i]-f_extended[i])/f_ts[i] for i in range(N)]
    # take the 2-norm of the errors
    norm_errors = myNorm(errors)
    # calculate a metric based on the l2-norm of the errors
    error_metric = T*pow(norm_errors,1/pow(T,3.0))
    
    # plot the segments
    if make_plot:
        # create a new plot with the segments overlaying the solutions
        plt.figure('flares')
        # plot the interpolated time-spectral solution
        plt.plot(t,f_ts,'ko')
        t_int,f_ts_int, dummy1 = fourierInterp(t,f_ts)
        plt.plot(t_int, f_ts_int,'k--', \
                 label=r'$time-spectral  \, \left(N='+str(N)+r'\right)$')
        # plot the time-marched segments
        for i in range(N):
            # while plotting the last segment, add a legend label
            if i == N-1:
                plt.plot(t_segments[i], f_segments[i],'g-', label=r'$time-marching$')
            else:
                plt.plot(t_segments[i], f_segments[i],'g-')
            # at the end of each segment, draw a star
            plt.plot(t_segments[i][-1], f_segments[i][-1], 'g*')
        #plt.rc('text', usetex=True)               # for using latex
        #plt.rc('font', family='serif')            # setting font
        #matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{mathtools}"]
        plt.xlabel(r'$t$', fontsize=18)
        plt.ylabel(r'$f(t)$', fontsize=18)
        plt.title(r'$ \|error\|_2 = '+str(norm_errors)+r' \, ; \quad \Delta t_{march} = '+str(delta_t)+r'$')
        plt.legend(loc='best')
            
    return(error_metric)

# this function returns the value of the definite integral using Simpson's Rule
def mySimpsonsRule(x, f, x_start, x_end, n_intervals, verbose=False):
    """
    This function approximates the definite integral of f(x) from x_start to 
    x_end using the composite Simpson's rule over a given, even number of  
    subintervals, n_intervals. The given abscissas, x, and ordinates, f, are
    linearly interpolated onto a new set of abscissas that span the integration
    range using the desired number of intervals. This interpolation is what is 
    finally integrated. This function implements the equation for the composite 
    Simpson's Rule found here:
    
    https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson.27s_rule
    
    Input:
      - abscissas over which the function f is defined, x (as a list)
      - the function to be integrated, f (as a list)
      - starting limit of integration, x_start
      - ending limit of integration, x_end
      - even number of subintervals to be used to span the limits, n_intervals

    Output:
      - value of the definite integral, integral
    """
    
    # raise an error if n_intervals is odd
    if n_intervals%2 == 1:
        raise ValueError('n_intervals must be even! ' + \
                         '\n\tn_intervals = %d)' % n_intervals)
    
    # raise an error if the limits of integration are beyond given abscissas
    if min(x)-x_start>1e-10 or x_end-max(x)>1e-10:
        raise ValueError('given limits of integration fall outside the ' +\
                         'range of abscissas provided.' + \
                         '\n\trange of abscissas: [' + str(min(x)) + ', ' + \
                         + str(max(x))+']' + \
                         '\n\tx_start = ' + str(x_start) + \
                         '\n\tx_end = ' + str(x_end))
    
    # compute the length of each interval
    h = (x_end-x_start)/n_intervals
    
    # abscissas for integration
    x_int = [x_start+i*h for i in range(n_intervals+1)]
    
    # if given abscissas do not already span the integration limits with 
    # n_intervals, then linearly interpolate the function onto new abscissas
    if x != x_int:
        x_int, f_int = linearInterp(x, f, x_int, verbose=False)
        if verbose:
            print('linear interpolation: simpson\'s rule')
    else:
        f_int = f
    
    # initial and final point contributions
    integral = f_int[0] + f_int[-1]
    
    # contributions from the remaining points
    for index in range(1,n_intervals):
        if index%2 == 1:
            # index is odd
            integral += 4.0*f_int[index]
        else:
            # index is even
            integral += 2.0*f_int[index]
    
    # multiply sum by h/3
    integral *= h/3.0

    return integral
        
# this function takes a TS solution, extends the last instance, integrates both
def integrate_segments(t, f_ts, T, make_plot=False, verbose=False):
    """
    This function takes a time-spectral solution, f_ts, computed assuming a 
    period T, and time-marches the last time instance forward by T, creating a 
    single, time-marched "flare," f_flare. Then, the time-spectral solution is 
    Fourier interpolated, f_ts_int, onto a grid that matches the number of 
    time-accurate steps used to march the final time instance up by one T. The 
    minimum value, f_min, seen between the time-spectral solution and the time-
    accurate one is recorded. The relative integration duration, t_span, is
    defined as the time between the start of the time trace and the last time 
    instance (i.e. t_span = the time at the last time instance). Then, two 
    integrals are taken:
    (1) f_ts_int - f_min from t=0 to t=t_span
    (2) f_flare - f_min from t=T to t=T+t_span
    N.B. Both f_ts_int and f_flare are first linearly interpolated onto their
    respective integration domains using an identical, odd number of points.
    (An odd number of points yields an even number of intervals, which can be 
    integrated with Simpson's Rule function.)
    The percent error between the two curves is returned as the error metric.
    
    Input:
      - the time instances, t (as a list)
      - time-spectral solution, f_ts (as a list)
      - the assumed period, T
      - whether or not to plot the segments, make_plot (defaults to False)
      
    Output:
      - some function of the l2-norm of the errors found at the time instances
      - if make_plot is set to True, then an open, unsaved figure is created
    """
    from math import ceil                    # for setting span_points
    import matplotlib                        # import by itself first
    matplotlib.use('Agg')                    # use Anti-Grain Geometry backend
    from matplotlib import pyplot as plt      # must be called AFTER use()
    
    # number of time instances
    N = len(t)
    
    # time-marching parameters
    points_per_T = 200
    
    # time step for time marching
    delta_t = T/(points_per_T-1)
    
    # set initial condition equal to the (t,f) of the last time instance
    t_start = t[N-1]              # initial condition, time
    initial_value = f_ts[N-1]     # intitial condition, value
    # intitalize lists for the time-marched segment
    t_segment = []
    f_segment = []
    # time stepping
    for n in range(points_per_T):
        t_segment.append(t_start+n*delta_t)
        if n == 0:
            # record the initial condition
            f_segment.append(initial_value)
        else:
            # explicitly step forward in time 
            f_segment.append(f_segment[n-1] + delta_t*myPeriodicODE(t_segment[n-1],f_segment[n-1]))
    
    # discretize [0,T] with equally spaced points_per_T
    t_int = myLinspace(0.0, T, points_per_T)
    # Fourier interpolate the time-spectral solution onto t_int
    t_int, f_ts_int, dummy1 = fourierInterp(t, f_ts, t_int)
    
    # define t_span as t[N-1] (time between 0 and the last time instance)
    t_span = t[N-1]
    # find the approx. number of points between 0 and t_span
    span_points = int(ceil((t_span/T)*(points_per_T/10.0)))
    
    # create a time grid from 0 to t_span
    t_ts_curve = myLinspace(0.0, t_span, span_points)
    # linearly interpolate the interpolated TS solution onto t_ts_curve
    t_ts_curve, f_ts_curve = linearInterp(t_int, f_ts_int, t_ts_curve, verbose=False)
    if verbose:
        print('linear interpolation: TS curve')
    
    # create a time grid from T to T+t_span
    t_ta_curve = myLinspace(T, T+t_span, span_points)
    # linearly interpolate the time-accurate segment from T to T+t_span
    t_ta_curve, f_ta_curve = linearInterp(t_segment, f_segment, t_ta_curve, verbose=False)
    if verbose:
        print('linear interpolation: TA curve')
    
    # find the minimum function value between the TS and time-accurate curves
    f_min = min(f_ts_int + f_segment)
    
    # integrate the time-spectral curve
    integrand_curve = [f_point-f_min for f_point in f_ts_curve]
    ts_area = mySimpsonsRule(t_ts_curve, integrand_curve, 0.0, t_span, span_points-1, verbose=False)
    # integrate the time-accurate curve
    integrand_curve = [f_point-f_min for f_point in f_ta_curve]
    ta_area = mySimpsonsRule(t_ta_curve, integrand_curve, T, T+t_span, span_points-1, verbose=False)
    
    # compute percent error of the area under the time-accurate segment from  
    # the area under the time-spectral curve
    percent_error = (ts_area-ta_area)*100/ts_area
    
    # set the error metric as the percent error
    error_metric = T/abs(percent_error/100)
    
    # plot the segment and integrated areas
    if make_plot:
        # create a new plot with the segments overlaying the solutions
        plt.figure('integrated_flare')       
        # plot the time spectral solution at the time instances
        plt.plot(t, f_ts, 'ko')
        # plot the interpolated time-spectral solution
        plt.plot(t_int, f_ts_int,'k--', \
                 label=r'$time-spectral \, \left(N='+str(N)+r'\right)$')
        # plot the time-marched segment
        plt.plot(t_segment, f_segment,'g-', label=r'$time-marching$')
        # at the end of the segment, draw a star
        plt.plot(t_segment[-1], f_segment[-1], 'g*')
        # create a line at the minimum value to [0,t_span]
        f_min_curve = [f_min]*span_points
        # fill in the space between f_min and the integrated TS curve
        plt.fill_between(t_ts_curve, f_ts_curve, f_min_curve, facecolor='magenta')
        # fill in the space between f_min and the integrated TA curve
        plt.fill_between(t_ta_curve, f_ta_curve, f_min_curve, facecolor='green')
        #plt.rc('text', usetex=True)               # for using latex
        #plt.rc('font', family='serif')            # setting font
        #matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{mathtools}"]
        plt.xlabel(r'$t$', fontsize=18)
        plt.ylabel(r'$f(t)$', fontsize=18)
        #plt.title(r'$ % error = '+str(error_metric)+r' \, ; \quad \Delta t_{march} = '+str(delta_t)+r'$')
        plt.legend(loc='best')
        plt.title('$Area_{TS} ='+str(round(ts_area,2))+' \, ; \quad Area_{TA} = '+str(round(ta_area,2))+' \, ; \quad \%_{error} ='+str(round(percent_error,2))+'$')
    
    return error_metric
    
# this function takes a TS solution, time-marches the last instance up a period
def extend_last_instance(t, f_ts, T, make_plot=False):
    """
    Given a time-spectral solution, this function takes the value at the last 
    time instance and treats it as the intial condition for a time-accurate 
    problem. This value is then explicitly advanced in time using Euler's 
    method by a duration of T. The values along the time-marched segment that 
    sit exactly T ahead of each original time instance are compared against the
    value seen at the time instance. The function returns some function of the 
    l2-norm of the differences.
    
    Input:
      - the time instances, t (as a list)
      - time-spectral solution, f_ts (as a list)
      - the assumed period, T
      - whether or not to plot the segments, make_plot (defaults to False)
      
    Output:
      - some function of the l2-norm of the errors found at the time instances
      - if make_plot is set to True, then an open, unsaved figure is created
    """
    import math
    import matplotlib                        # import by itself first
    matplotlib.use('Agg')                    # use Anti-Grain Geometry backend
    from matplotlib import pylab as plt      # must be called AFTER use()
    
    # number of time instances
    N = len(t)
    
    # factor by which the TS grid (the time instances) is refined to give the
    # time-accurate grid for time-marching the last point. (if there are 9 time
    # instaces that are just shy of spanning the time-spectral curve, and if 
    # the refinement factor is 10, then there will be 90 points that are just
    # shy of spanning the same period, over which time-marching will occur.)
    refinement_fac = 20
    
    # fine time grid spanning a period (including both end points)
    points_per_T = refinement_fac*N+1
    
    # time step for time marching
    delta_t = T/(points_per_T-1)
    
    # set initial condition equal to the (t,f) of the last time instance
    t_start = t[N-1]              # initial condition, time
    initial_value = f_ts[N-1]     # intitial condition, value
    # intitalize lists for the time-marched segment
    t_segment = []
    f_segment = []
    # time stepping
    for n in range(points_per_T):
        t_segment.append(t_start+n*delta_t)
        if n == 0:
            # record the initial condition
            f_segment.append(initial_value)
        else:
            # explicitly step forward in time 
            f_segment.append(f_segment[n-1] + delta_t*myPeriodicODE(t_segment[n-1],f_segment[n-1]))
    
    # extract the points that sit one T ahead of each time instance
    f_ta_compare = []
    t_ta_compare = []    
    for n in range(1,points_per_T):
        if n % refinement_fac == 0:
            t_ta_compare.append(t_segment[n])
            f_ta_compare.append(f_segment[n])
    
    # compute the error between the TS solution and the extended period
    errors = [abs(f_ts[i]-f_ta_compare[i])/f_ts[i] for i in range(N)]
    
    # extend only to the first two instances
    #errors = [abs(f_ts[i]-f_ta_compare[i])/f_ts[i] for i in range(1)]
    
    # look at only the difference and slope of the first two points
    diff_der_ts = (f_ts[1]-f_ts[0]) + (f_ts[1]-f_ts[0])/t[1]
    diff_der_ta = (f_ta_compare[1]-f_ta_compare[0]) + (f_ta_compare[1]-f_ta_compare[0])/t[1]
    #errors = [abs(diff_der_ts - diff_der_ta)]
    
    # compute the distances between the first time instances and all the others
    ts_distances = [math.sqrt( (i*t[1])**2 + (f_ts[i]-f_ts[0])**2 ) for i in range(1,N)]
    ta_distances = [math.sqrt( (i*t[1])**2 + (f_ta_compare[i]-f_ta_compare[0])**2 ) for i in range(1,N)]
    cum_dist_diff = sum(ta_distances) - sum(ts_distances)
    
    # take a TA step at each time instance and then move forward T, let that be
    # the new initial condition, and take another TA step. compare the results
    f_stepped = []
    f_plus_T_stepped = []
    stepped_differences = []
    for i in range(N):
        f_stepped.append(f_ts[i] + delta_t*myPeriodicODE(t[i],f_ts[i]))
        f_plus_T_stepped.append(f_ts[i] + delta_t*myPeriodicODE(t[i]+T,f_ts[i]))
        stepped_differences.append(abs(f_stepped[i]-f_plus_T_stepped[i]))
        
    # take the 2-norm of the errors
    norm_errors = myNorm(errors)
    # add up the errors
    sum_errors = sum(errors)
    
    # calculate a metric based on the l2-norm of the errors
    error_metric = T*pow(norm_errors,1/pow(T,3.0))
    error_metric = norm_errors
    error_metric = sum_errors
    error_metric = abs(cum_dist_diff) + norm_errors
    error_metric = sum(stepped_differences)
            
    # plot the segments
    ta_method = 'T_&_test'
    if make_plot:
        # create a new plot with the segments overlaying the solutions
        plt.figure('last_flare')
        # plot the interpolated time-spectral solution
        plt.plot(t,f_ts,'ko')
        t_int,f_ts_int, dummy1 = fourierInterp(t,f_ts)
        plt.plot(t_int, f_ts_int,'k--', \
                 label=r'$time-spectral  \, \left(N='+str(N)+r'\right)$')
        if ta_method == 'last_flare':
            # plot the time-marched segment
            plt.plot(t_segment, f_segment,'g-', label=r'$time-marching$')
            # plot time-accurate points that sit T ahead of the time instances
            plt.plot(t_ta_compare, f_ta_compare, 'go')
        if ta_method == 'T_&_test':
            # plot the time-marched portions on the TS solution
            for i in range(N):
                if i == 0:
                    plt.plot([t[i],t[i]+delta_t],[f_ts[i],f_stepped[i]],'c-',label=r'$time-marching$')
                else:
                    plt.plot([t[i],t[i]+delta_t],[f_ts[i],f_stepped[i]],'c-')
            # plot the time-marched bits that lie T ahead of the TS solution
            for i in range(N):
                plt.plot([t[i]+T,t[i]+T+delta_t],[f_ts[i],f_plus_T_stepped[i]],'g-')
            # plot horizontal lines from the original time instances up a T
            for i in range(N):
                plt.plot([t[i],t[i]+T],[f_ts[i]]*2,'y--')
        #plt.rc('text', usetex=True)               # for using latex
        #plt.rc('font', family='serif')            # setting font
        #matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{mathtools}"]
        plt.xlabel(r'$t$', fontsize=18)
        plt.ylabel(r'$f(t)$', fontsize=18)
        plt.title(r'$ error \, metric = '+str(round(error_metric,8))+r' \, ; \quad \Delta t_{march} = '+str(round(delta_t,6))+r'$')
        plt.legend(loc='best')
        
    return(error_metric)
 
# main ########################################################################
def main():
    
    import math
    import matplotlib                        # import by itself first
    matplotlib.use('Agg')                    # use Anti-Grain Geometry backend
    from matplotlib import pyplot as plt      # must be called AFTER use()
    from matplotlib import animation         # for specifying the writer
    
    # close open windows
    plt.close('all')
    
    # user inputs
    N = 17                  # number of time instaces
    T = 2*math.pi          # period of osciallation (enter a float!)
    T = 2.0 #not really being used right now...10/29/15
    
    # create the time-spectral operator 
    D = time_spectral_operator(N,T)
    
    # time interval
    delta_T = float(T)/N
         
    # indices corresponding to the time instances
    indices = list(range(N))
         
    # list of time instances
    t = [delta_T*index for index in indices]
    
    ###########################################################################
    # Check to see if the time-spectral operator is working ###################
    ###########################################################################
    
    # sampling at the time instances
    f_TS, dummy1, dummy2, name = myPeriodicSignal(t,T)
    
    # the time derivatives at the time instances using time-spectral operator
    dfdt_TS = myMult(D, f_TS)
    
    # find the second derviative using the time-spectral operator as well
    df2dt2_TS = myMult(D, dfdt_TS)
    
    # USER INPUT: set the type of interpolation (fourier, linear)
    interp_type = 'fourier'
    
    if interp_type == 'fourier':
        # interpolate the time-spectral results with Fourier series
        t_int, dfdt_TS_int, dummy1 = fourierInterp(t, dfdt_TS)
        t_int, df2dt2_TS_int, dummy1 = fourierInterp(t, df2dt2_TS)
        interp_label = '$(Fourier \, Interp.)$'
    
    if interp_type == 'linear':
        # interpolate the time-spectral results with line segments
        t_int, dfdt_TS_int = linearInterp(t, dfdt_TS)
        t_int, df2dt2_TS_int = linearInterp(t, df2dt2_TS)
        interp_label = '$(Linear \, Interp.)$'
    
    # fine time grid (10 times the number of time instances) for "exact" values
    t_fine = [T*index/(10*N-1) for index in range(10*N)]
    f_fine, dfdt_fine, df2dt2_fine, name = myPeriodicSignal(t_fine,T)
    
    # plotting: USER INPUT. Would you like to plot this verification figure?
    plot_figure = False
    plot_name = 'TS verification'
    
    if plot_figure == True:
        # plotting
        print('\nplotting fig. ' + plot_name + '...')
        plt.rc('text', usetex=True)               # for using latex
        plt.rc('font', family='serif')            # setting font
        # plot the "exact" results
        plt.plot(t_fine,f_fine,'k-',label='$f$')
        plt.plot(t_fine,dfdt_fine,'r-',label='$df/dt$')
        plt.plot(t_fine,df2dt2_fine,'b-',label='$d^2f/dt^2$')
        # plot the time instances
        plt.plot(t,f_TS,'ko',label='$f_{TS}$')
        # plot the time-spectral first dervative and interpolation
        plt.plot(t,dfdt_TS,'go',label='$df_{TS}/dt$')
        plt.plot(t_int,dfdt_TS_int,'g--',label=interp_label)
        # plot the time-spectral second dervative and interpolation
        plt.plot(t,df2dt2_TS,'yo',label='$d^2f_{TS}/dt^2$')
        plt.plot(t_int,df2dt2_TS_int,'y--',label=interp_label)     
        # limits, labels, legend, title
        plt.xlabel(r'$t$', fontsize=18)
        plt.ylabel('')
        plt.legend(loc='best', ncol=2)
        #plt.title(r'$N = \,$'+str(N))
        plt.title(r'$'+name+r'$ \quad (N = \,$'+str(N)+'$)$')
        # save figure
        print('saving fig. ' + plot_name + '...')
        plt.savefig(plot_name, dpi=500)
        print('fig.' + plot_name + ' saved')
        plt.close()
    
    ###########################################################################
    # [time accurate] explicit euler ##########################################
    ###########################################################################
    delta_t = 0.05             # time step
    initial_value = 8.0        # intitial condition
    #initial_value = 8.905      # steady-state peak
    #initial_value = 8.841      # steady-state trough
    #initial_value = 8.82
    #initial_value = 8.71
    t_start = 0                # initial time
    t_end = 60                 # approx. final time (stop at or just after)
  
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
            f.append(f[n-1] + delta_t*myPeriodicODE(times[n-1],f[n-1]))
            
    # plotting: USER INPUTS! do you want to animate the solution history or just
    # plot the final result? (True = animate, False = just print final result)
    animate_plot = False                    
    plot_name = 'time-accurate ODE'
    n_images = time_points            # total number of images computed
    skip_images = 3                   # images to skip between animation frames
    
    # plotting: initializations
    fig = plt.figure()
    l, = plt.plot([], [],'k-',label='f')

    # plotting: things that will not be changing inside the loop
    plt.rc('text', usetex=True)               # for using latex
    plt.rc('font', family='serif')            # setting font
    plt.xlabel(r'$t$', fontsize=18)
    plt.ylabel(r'$f(t)$', fontsize=18)
    plt.xlim(0,t_end)
    vertical_padding = (max(f)-min(f))/4.0
    plt.ylim(min(f)-vertical_padding,max(f)+vertical_padding)
    plt.title(r'$\Delta t = \,$'+str(delta_t))
    
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
            plt.plot(times[:n+1],f[:n+1])
            # progress monitor
            percent_done = float(n)*100.0/(n_images-1)
            print('capturing fig. '+plot_name+' (frame #'+str(frame)+'): ', \
                   percent_done,'%')
            writer.grab_frame()
            frame += 1
        writer.grab_frame()
    
    # plotting: save an image of the final frame
    print('\n'+'saving final image...')
    plt.savefig(plot_name, dpi=500)
    print('figure saved: ' + plot_name)
    
    # free memory used for the plot
    plt.close(fig)
       
    ###########################################################################
    # [time accurate] extract period from time-marching solution ##############
    ###########################################################################
    
    # pass the time-accurate solution history to the extractPeriod function
    T_extracted, t_period, f_period = extractPeriod(times,f)
    
    # plot an isolated period by itself
    plt.figure()
    plt.plot(t_period,f_period,'k-')
    #plt.rc('text', usetex=True)               # for using latex
    #plt.rc('font', family='serif')            # setting font
    #matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{mathtools}"]
    plt.xlabel(r'$t$', fontsize=18)
    plt.ylabel(r'$f(t)$', fontsize=18)
    plt.title(r'$steady-state \, period \, extracted \, from \, '+\
                r'time-accurate \, result$')
    plot_name = 'isolated period extracted'
    print('saving image...')
    plt.savefig(plot_name, dpi=300)
    print('figure saved: ' + plot_name)
    plt.close()
      
    ###########################################################################
    # [time spectral] explict pseudo-timestepping (dfdt -> f) #################
    ###########################################################################
    
    # MESSING WITH THE PERIOD (and, therefore the operator matrix!!!)
    # (Note that when the period is wrong, so are the associated time instances)
    wrong_period = True # (only used for plotting)
    T_actual = 2.0      # (only used for plotting)
    adjust_period = True # true if the period is being changed iteratively
    gopinath_method = False    # use Gopinath's method to adjust the period
    compare_to_bdf = False # finite difference the TS solution and compare
    extension_method = False    # use the instance-extention method to adjust T
    integration_method = False # use the last-segment integration method
    last_extension_method = True # method of only extending the last instance
    conv_extend_start = 1e-1  # level of convergence to start extension method
     
    
    T = 60.5
    
    
    # switch on the period plotting (has to be set here)
    if adjust_period:    
        period_plot = True            
        period_found = False
    else:
        period_plot = False
        
    # create the time-spectral operator 
    D = time_spectral_operator(N,T)    
    # time interval
    delta_T = float(T)/N         
    # indices corresponding to the time instances
    indices = list(range(N))         
    # list of time instances
    t = [delta_T*index for index in indices]
    
    # create the 2nd-order finite differncing matrix (for residual correction)
    D_bdf = [[0]*N for i in range(N)]
    for i in range(N):
        for j in range(N):
            if i == 0:
                D_bdf[0][0] = 3.0
                D_bdf[0][-2:] = [1.0, -4.0]
            if i == 1:
                D_bdf[1][:2] = [-4.0, 3.0]
                D_bdf[1][-1] = 1.0
            if i >= 2:
                D_bdf[i][i-2] = 1.0
                D_bdf[i][i-1] = -4.0
                D_bdf[i][i] = 3.0
    for i in range(N):
        for j in range(N):
            D_bdf[i][j] = D_bdf[i][j]/(2.0*delta_T)
    
    # pseudo-timestep size (residual)
    adjust_delta_tau = True      # allow pseudo-time to change during run
    delta_tau_init = 0.00001
    #delta_tau = 0.00191            # pseudo-timestep (optimal for guess=100)
    delta_tau = delta_tau_init
    # record pseudo-time step
    delta_tau_hist = []
    delta_tau_hist.append(delta_tau)
    going_up_count = 0
    
    # pseudo-timestep size (for the period!)
    delta_tau_T = 0.0001    # pseudo-timestep for driving the period
    
    init_value = 8.91            # constant initial guess
    max_pseudosteps = 5000000     # maximum number of pseudo-timesteps to try
    
    conv_criteria = 1e-4          # resdiual convergence criteria
    
    # uncomment to use machine zero as the convergence criterion
    #import sys                               # for getting machine zero value
    #machine_zero = sys.float_info.epsilon
    #conv_criteria = my_machine_zero         # resdiual convergence criteria
    
    
    
    
    
    # Initialize history lists
    res_hist = []                # residual history
    res_bdf_hist = []            # residual history for the bdf result
    f_TS_hist =[]                # solution history
    
    if adjust_period:
        T_hist = []                  # period history
        error_metric_hist = []       # error metric history
        # append the intital guess of the period
        T_init = T
        T_hist.append(T_init)
 
    # set the intial guess for the periodic solution
    f_TS = [init_value for index in indices]    # N.B. f_TS is being reassigned
    f_TS_hist.append(f_TS)
    
    old_error_metric = 1e10               # intitialize to something really big
    
    # compute the starting error metric for the TA-corrected methods
    if period_plot:
        if extension_method:
            error_metric = extend_instances(t, f_TS, T)
        if integration_method:
            error_metric = integrate_segments(t, f_TS, T)
            metric_increasing = False    # needs to be initialized
        if last_extension_method:
            error_metric = extend_last_instance(t, f_TS, T)
            prime_counter = 0
            # list of primes from 2 to 101
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101]
            primes = [2, 3, 5, 7, 11]
            primes = [2, 3, 5]
            T_candidate = 'none yet'        
        # append error metric to history 
        error_metric_hist.append(error_metric)
    
    # pseudo-timestepping
    for iteration in range(max_pseudosteps):
        
        
            
        
        
        
        # compute dfdt from the ODE
        dfdt = [myPeriodicODE(t[index],f_TS[index]) for index in indices]
        # compute D*f
        Df = myMult(D,f_TS)
        # compute the residual vector for this solution
        res = [-Df[index]+dfdt[index] for index in indices]
        # find the norm of the residual vector and store it
        res_hist.append(myNorm(res))
        # check residual stopping condition
        if res_hist[iteration] < conv_criteria:
            print("\nsolution found.", iteration, " iterations required.\n") 
            break
        
        
        
        # check residual stopping condition
        if res_hist[iteration] < conv_criteria:
            print("\nsolution found.", iteration, " iterations required.\n") 
            break
        else:
            # adjust the pseudo-timestep
            if adjust_delta_tau == True and iteration > 1:
                current_res = res_hist[iteration]
                previous_res = res_hist[iteration-2]
                ratio = previous_res/current_res                      
                scaling_fac = 1.001
                delta_tau_new = delta_tau*scaling_fac*ratio
                #delta_tau = delta_tau_new
                
                # switch to initial value
                if ratio < 1.0:
                    if going_up_count == 0:
                        unstable_iter = iteration
                    going_up_count += 1
                    print('\n\nRESIDUAL GOING UP!')
                    unstable_iter -= 1
                    if unstable_iter > 1 and going_up_count <=2:
                        delta_tau = delta_tau_hist[unstable_iter-1]
                    else:
                        if delta_tau/2 > delta_tau_init:
                            delta_tau = delta_tau/2
                        else:
                            delta_tau = delta_tau_init
                        
                else:
                    going_up_count = 0
                    delta_tau = delta_tau_new
                    
                
                # limit slope 
                #delta_tau_slope = (delta_tau_new-delta_tau)/1.0   # 1 sweep
                #if delta_tau_slope > 1.0:
                #    delta_tau = delta_tau
                #else:
                #    delta_tau = delta_tau_new
                
            # record time step used (keep it outside the if statment so that it
            # gets recorded at every iteration)
            delta_tau_hist.append(delta_tau)
        
        
        
        
        
        
        
        # compare to a differention of the time instances
        if compare_to_bdf:
            # compute D_bdf*f
            Df_bdf = myMult(D_bdf,f_TS)
            # compute the BDF residual vector for this solution
            res_bdf = [-Df_bdf[index]+dfdt[index] for index in indices]
            # find the norm of the BDF residual vector and store it
            res_bdf_hist.append(myNorm(res_bdf))
                
        # print new residual to the screen
        if adjust_period == True:
            print('pseudo-time iteration #',iteration, \
                  '; residual = '+str(res_hist[iteration]), \
                  '; T = ', T, \
                  '; d_tau = ', delta_tau)
            if last_extension_method:
                if prime_counter > 0:
                    print('prime = '+str(primes[prime_counter-1]), \
                    '; T_candidate = '+str(T_candidate))
        elif compare_to_bdf:
            print('pseudo-time iteration #',iteration, \
                  '; residual = '+str(res_hist[iteration]), \
                  '; BDF residual = '+str(res_bdf_hist[iteration]))
        else:
            if adjust_delta_tau:
                print('pseudo-time iteration #',iteration, \
                  '; residual = '+str(res_hist[iteration]), \
                  '; d_tau = ', delta_tau)
            else:
                print('pseudo-time iteration #',iteration, \
                  '; residual = '+str(res_hist[iteration]))
        
        
        
        # compute the correct error metric for the TA-corrected methods
        if period_plot:
            if extension_method:
                error_metric = extend_instances(t, f_TS, T)
            if integration_method:
                error_metric = integrate_segments(t, f_TS, T)
            if last_extension_method:
                error_metric = extend_last_instance(t, f_TS, T)
            
        # if adjusting period
        if adjust_period == True:
            
            # Gopinath's method
            if gopinath_method:
                # GOPINATH: compute "figure of merit" (NOT the residual)
                # N.B. Here, R from the paper is -df/dt
                I = [Df[index]-dfdt[index] for index in indices]
                # compute gradient of the square of the "figure of merit"
                grad_I2_wrt_T = [(-2.0/T)*I[index]*Df[index] for index in indices]
                # compute the average of these gradients
                ave_grad_I2_wrt_T = sum(grad_I2_wrt_T)/float(len(grad_I2_wrt_T))
                # using the same pseudo timestep as for the solution vector
                T = T - delta_tau_T*ave_grad_I2_wrt_T
            
            # extension methods
            if res_hist[iteration] < conv_extend_start:
                
                if extension_method:
                    #error_metric = extend_instances(t, f_TS, T)
                    print('error_metric = ', error_metric)
                    if old_error_metric-error_metric >= 0:
                        print('; error metric: DOWN \n\n')
                        T = T - delta_tau_T*error_metric
                    else:
                        print('error metric: UP\n\n')
                        adjust_period = False
                    old_error_metric = error_metric
                
                if integration_method:
                    print('error metric = ', error_metric)
                    # as the slope starts going up, reduce the step size
                    T = T + delta_tau_T
                    if old_error_metric < error_metric/2.0:
                        metric_increasing = True
                        print('; error metric: INCREASING \n\n')
                    if metric_increasing and old_error_metric < error_metric/100:
                        adjust_period = False
                    old_error_metric = error_metric
                
                if last_extension_method:
                    
                    
                    if error_metric < 1e-7:

                        # if the error metric is basically zero, then we've 
                        # found a candidate solution for T
                        print('Candidate T found!')
                        T_candidate = T
                        
                        # set the prime counter to the first index
                        #prime_counter = 0
                        
                        # start the checking process by dividing the T in half
                        if prime_counter != len(primes):
                            T = T_candidate/primes[prime_counter]
                            prime_counter += 1
                        else:
                            # this is the case we have found a new candidate
                            # solution, but have no more primes left to check 
                            # it with. since we've already checked all of our
                            # primes before getting here, it's probably the 
                            # answer
                            T = T_candidate
                            period_found = True
                        
                    else:
                        
                        # if the error metric isn't zero, then we need a new T
                        if prime_counter == len(primes):
                            # we've gotten to the end of the list of primes, 
                            # then our candidate solution is our final solution
                            T_candidate = T*primes[-1]
                            T = T_candidate
                            period_found = True
                        elif prime_counter != 0:
                            # if we're still in the process of checking a 
                            # candidate solution and we hadn't yet reached the 
                            # last prime, then move on to the next prime
                            T_candidate = T*primes[prime_counter-1]
                            T = T_candidate/primes[prime_counter]                            
                            print('checking...prime='+str(primes[prime_counter]))
                            prime_counter += 1
                        else:
                            # if we get here, it means no candidate solution 
                            # has been found. reduce the period by delta_tau_T
                            T = T - delta_tau_T
                            
                        
                        
                    #error_metric = extend_instances(t, f_TS, T)
                    #print('error_metric = ', error_metric)
                    #if error_metric > 0.00001 and T > 1.95:
                    #    T = T - delta_tau_T
                    
                    #if old_error_metric-error_metric >= 0:
                    #    print('; error metric: DOWN \n\n')
                    #    T = T - delta_tau_T*error_metric
                    
                    #else:
                        #print('error metric: UP\n\n')
                    #    adjust_period = False
                    #old_error_metric = error_metric
                
                
        # if adjusting period, recompute the matrix and the time instances
        if adjust_period == True:
            # create the time-spectral operator 
            D = time_spectral_operator(N,T)    
            # time interval
            delta_T = float(T)/N         
            # indices corresponding to the time instances
            indices = list(range(N))         
            # list of time instances
            t = [delta_T*index for index in indices]
            # turn off the period finding if we've found the period
            if period_found:
                adjust_period = False
            
        # append this to the period history
        T_hist.append(T)
        
        # record the error metric for plotting - DON'T RECOMPUTE!
        #if extension_method:
        #    error_metric = extend_instances(t, f_TS, T)
        #if integration_method:
        #    error_metric = integrate_segments(t, f_TS, T)
        error_metric_hist.append(error_metric)
            
        # update solution vector
        f_TS = [f_TS[index]+delta_tau*res[index] for index in indices]
        # store new solution for plotting
        f_TS_hist.append(f_TS)
        
    
        
    # plotting: user input! do you want to animate the solution history or just
    # plot the final result? (True = animate, False = just print final result)
    animate_plot = True                  
    plot_name = 'TS explicit pseudo'
    n_images = iteration+1
    skip_images = 7000
    
    # plotting: initializations and sizing
    if period_plot:
        n_plots = 3        
        stretch = 2.5    #. N.B. THIS IS THE HIGHEST THIS CAN BE!!!
    else:
        n_plots = 2
        stretch = 2.0
    fig = plt.figure()
    xdim, ydim = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches(stretch*xdim, ydim, forward=True)
    
    # plotting: things that will not be changing inside the loop
    #plt.rc('text', usetex=True)               # for using latex
    #plt.rc('font', family='serif')            # setting font
    # solution history plot
    plt.subplot(1,n_plots,1)
    #dots, = plt.plot([],[],'ko',label='$f_{TS}$')
    #line, = plt.plot([],[],'k--',label='$ Fourier \, interp.$')
    
    #plt.legend(loc='upper right')
    #ave_init_value = sum(f_TS_hist[0])/N
    max_final = max(f_TS_hist[-1])
    min_final = min(f_TS_hist[-1])
    ampl_final = abs(max_final-min_final)
    white_space = ampl_final/3.0
    #plt.xlim(0,T)
    #if animate_plot == True:
    #    start = min(ave_init_value, max_final+white_space)
   #     finish = max(ave_init_value, max_final+white_space)        
        #plt.ylim(start, finish)
   # else:
   #     start = min(min_final-white_space,max_final+white_space)
   #     finish = max(min_final-white_space,max_final+white_space)
   #     plt.ylim(start, finish)
    # residual history plot
    plt.subplot(1,n_plots,2)
    #res, = plt.semilogy([],[],'b-',label='TS residual')
    #bdf_res, = plt.semilogy([],[],'g-',label='BDF residual')
    plt.xlabel(r'$iteration$', fontsize=18)
    plt.ylabel(r'$\|R\|$', fontsize=18)
    plt.xlim(0,iteration)
    min_power = int(math.log(min(res_hist),10))-1
    max_power = int(math.log(max(res_hist),10))+1
    plt.ylim(pow(10,min_power), pow(10,max_power))
    # period history plot
    if period_plot:
        plt.subplot(1,n_plots,3)
        #per, = plt.plot([],[],'r-')
        plt.plot(list(range(iteration)),[T_actual]*iteration,'k--',label='$T_{actual}$')
        plt.xlabel(r'$iteration$', fontsize=18)
        plt.ylabel(r'$T$', fontsize=18)
        plt.xlim(0,iteration)
        space = (max(T_hist)-min(T_hist))/3.0
        plt.ylim(min(T_hist)-space,max(T_hist)+space)
        plt.title(r'$\Delta\tau_T = '+str(delta_tau_T)+'$')
    # plotting: set the total number of frames
    if animate_plot == True:
        # capture all frames (skipping, if necessary) and the final frame
        all_frames = list(range(0,n_images,skip_images+1))+[n_images-1]
    else:
        # no animation: just capture the last one
        all_frames = [n_images-1]
        
    # plotting: capturing the movie
    writer = animation.writers['ffmpeg'](fps=10)
    with writer.saving(fig, plot_name+'.mp4', 100):
        frame = 0
        for n in all_frames:
            # plot solution and interpolation
            plt.subplot(1,n_plots,1)
            plt.cla()
            t_plot = [(float(T_hist[n])/N)*index for index in indices]
            plt.plot(t_plot,f_TS_hist[n],'ko',label='$f_{TS}$')
            t_int,f_TS_int, dummy1 = fourierInterp(t_plot,f_TS_hist[n])
            plt.plot(t_int,f_TS_int,'k--',label='$ Fourier \, interp.$')
            plt.xlabel(r'$t$', fontsize=18)
            plt.ylabel(r'$f\left(t\right)$', fontsize=18)
            plt.xlim(0,T_hist[n])
            plt.ylim(min(f_TS_hist[n])-white_space,max(f_TS_hist[n])+white_space)
            if period_plot:
                plt.subplot(1,n_plots,1)
                plt.xlim(0,max(T_hist))
            plt.title(r'$iteration \,\#$'+'$'+str(n)+'$')
            #plt.legend(loc='best')
            # plot residual            
            plt.subplot(1,n_plots,2)
            if n > 0 and res_hist[n] >= res_hist[0]:
                plt.semilogy(res_hist[:n+1],'g-')
            else:
                plt.semilogy(res_hist[:n+1],'r-',label='TS residual')
            plt.title(r'$\Delta\tau = '+str(delta_tau_hist[n])+'$')
            if compare_to_bdf:
                plt.plot(list(range(n)),res_bdf_hist[:n])
                plt.legend(loc='best')
            # plot period
            if period_plot:
                plt.subplot(1,n_plots,3)
                plt.plot(T_hist[:n+1],'m-')
                plt.xlim(0,iteration)
            # progress monitor
            frame += 1
            percent_done = float(n)*100.0/(n_images-1)
            print('capturing fig. '+plot_name+' (frame #'+str(frame)+'): ', \
                   percent_done,'%')
            writer.grab_frame()
        # for the last frame, adjust the x limits
        if period_plot:
                plt.subplot(1,n_plots,1)
                plt.ylim(min_final-white_space,max_final+white_space)
                plt.xlim(0,T_hist[-1])
        writer.grab_frame()
        
    # plotting: rescale the final frame to focus on the converged solution
    if animate_plot == True:
        plt.subplot(1,n_plots,1)
        plt.ylim(min_final-white_space,max_final+white_space)
        plt.xlim(0,T_hist[-1])
    
    # plotting: save an image of the final frame
    print('saving final image...')
    plt.savefig(plot_name, dpi=500)
    print('figure saved: ' + plot_name + '\n')
    
    # free memory used for the plot
    plt.close(fig)
        
    # print and plot information about the period-finding solution
    if period_plot:
        print('Period-finding process:')
        print('  T_init = '+str(T_init))
        print('  T_found = '+str(T)+'\n')

        fig = plt.figure()
        plot_name = 'error metric vs T'
        plt.semilogy(T_hist,error_metric_hist,'ko', markersize=2)
        plt.xlabel(r'$T$', fontsize=18)
        plt.ylabel(r'$error \,\, metric$', fontsize=18)
        plt.title(r'$\Delta\tau = '+str(delta_tau)+r'\, ; \, \Delta\tau_T = '+str(delta_tau_T)+'\, ; \, T_{found} = '+str(round(T,3))+'$')
        print('saving final image...')
        plt.savefig(plot_name, dpi=500)
        print('figure saved: ' + plot_name + '\n')
        plt.close(fig)
    
    # plot the variation of the pseudo-time step
    if adjust_delta_tau:
        fig = plt.figure()
        plot_name = 'pseudo-time step'
        plt.semilogy(list(range(iteration)),delta_tau_hist[:-1],'k.-')
        plt.xlabel(r'$iteration$', fontsize=18)
        plt.ylabel(r'$\Delta\tau$', fontsize=18)
        print('saving final image...')
        plt.savefig(plot_name, dpi=300)
        print('figure saved: ' + plot_name + '\n')
        plt.close(fig)
    ###########################################################################
    # compare the time-spectral results against the time-accurate one #########
    ###########################################################################
    
    # set the number of comparison points to that of the finer solution
    n_comp_pts = max(len(t_period),len(t_int))
    t_comp_pts = myLinspace(0,T,n_comp_pts)
    
    # interpolate both solutions onto the finer grid (interpolating the finer
    # solution will return the same thing...redundant, but easier to code)
    t_period_comp, f_period_comp = linearInterp(t_period, f_period, t_comp_pts)
    t_int_comp, f_TS_int_comp = linearInterp(t_int, f_TS_int, t_comp_pts)
    
    # now, try shifting the extracted TA period, one index at a time (from 3
    # indices before to 3 indices after its current location), until the sum of
    # the differences between it and the time-spectral curve is the smallest
    max_shift = 10
    trial_shifts = list(range(-max_shift, max_shift+1))
    norm_diffs = []
    for shift in trial_shifts:
        f_period_shift = f_period_comp[shift:] + f_period_comp[:shift]
        diffs = [f_period_shift[i]-f_TS_int_comp[i] for i in range(n_comp_pts)]
        norm_diffs.append(myNorm(diffs,1))
    
    # recover the number of shifted indices that minimize the difference
    min_index = norm_diffs.index(min(norm_diffs))
    optimal_shift = trial_shifts[min_index]
    f_period_opt = f_period_comp[optimal_shift:]+f_period_comp[:optimal_shift]
    
    # plot boths isolated periods atop one another
    plt.figure()
    # plot the extracted time-accurate period
    if wrong_period:
        plt.plot(t_period,f_period,'b-',label=r'$time-accurate$')
    else:
        plt.plot(t_int_comp,f_period_opt,'b-', label=r'$time-accurate$')
    # plot the interpolated time-spectral solution
    plt.plot(t,f_TS_hist[-1],'ko')
    plt.plot(t_int_comp, f_TS_int_comp,'k--', \
             label=r'$time-spectral \left(N='+str(N)+r'\right)$')
    #plt.rc('text', usetex=True)               # for using latex
    #plt.rc('font', family='serif')            # setting font
    #matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{mathtools}"]
    plt.xlabel(r'$t$', fontsize=18)
    plt.ylabel(r'$f(t)$', fontsize=18)
    #if wrong_period:
    #    plt.ylim(8.84,8.91)
    plt.legend(loc='best')
    plot_name = 'comparison - TS vs TA'
    print('saving image...')
    plt.savefig(plot_name, dpi=500)
    print('figure saved: ' + plot_name + '\n')
    plt.close()

    ###########################################################################
    # plot the extended segment(s) corresponding to the converged solution. ###
    ###########################################################################
    
    if extension_method:
        
        # compute the segments and start the plot
        dummy1 = extend_instances(t, f_TS_hist[-1], T, make_plot=True)
        
        # continue adding to the plot created in the above subroutine
        plt.figure('flares')
        # plot the extracted time-accurate period
        if wrong_period:
            plt.plot(t_period,f_period,'b-',label=r'$time-accurate$')
        else:
            plt.plot(t_int_comp,f_period_opt,'b-', label=r'$time-accurate$')
        # finish decorating the plot, save, close
        plt.legend(loc='best')
        plot_name = 'comparison - TS + marching vs TA '
        print('saving image...')
        plt.savefig(plot_name, dpi=500)
        print('figure saved: ' + plot_name + '\n')
        plt.close('flares')
        
    if integration_method:
        
        # compute the two integrals and start the plot
        dummy1 = integrate_segments(t, f_TS, T, make_plot=True)
        
        # continue adding to the plot created in the above subroutine
        plt.figure('integrated_flare')
        # plot the extracted time-accurate period
        if wrong_period:
            plt.plot(t_period,f_period,'b-',label=r'$time-accurate$')
        else:
            plt.plot(t_int_comp,f_period_opt,'b-', label=r'$time-accurate$')
        # finish decorating the plot, save, close
        plt.legend(loc='best')
        plot_name = 'comparison - TS + marched integration vs TA '
        print('saving image...')
        plt.savefig(plot_name, dpi=500)
        print('figure saved: ' + plot_name + '\n')
        plt.close('integrated_flare')
    
    if last_extension_method:
        
        # compute the segments and start the plot
        dummy1 = extend_last_instance(t, f_TS, T, make_plot=True)
        
        # continue adding to the plot created in the above subroutine
        plt.figure('last_flare')
        # plot the extracted time-accurate period
        if wrong_period:
            plt.plot(t_period,f_period,'b-',label=r'$time-accurate$')
        else:
            plt.plot(t_int_comp,f_period_opt,'b-', label=r'$time-accurate$')
        # finish decorating the plot, save, close
        plt.legend(loc='best')
        plot_name = 'comparison - TS + last_marched vs TA '
        print('saving image...')
        plt.savefig(plot_name, dpi=500)
        print('figure saved: ' + plot_name + '\n')
        plt.close('flares')




'''

    ###########################################################################
    # [time spectral] Gauss-Seidel analogy w/ variable pseudo-timestep ########
    ###########################################################################    
    import random
    
    adjust_delta_tau = False      # allow pseudo-time to change during run
    delta_tau_init = 0.0927       # pseudo-timestep (optimal for guess=100)
    delta_tau_init = 0.01
    
    max_sweeps = 1000000          # max number of Gauss-Seidel sweeps to try
    init_value = 100            # constant intital guess
    conv_criteria = 1e-4         # resdiual convergence criteria
    
    # Initialize lists
    res_hist = []                # residual history
    f_TS_hist = []               # solution history
    delta_tau_hist = []          # pseudo-timestep history
    f_TS_new = []                # new solution (needed only to stop warning)
    
    # begin the Gauss-Seidel sweeps. (N.B. One sweep corresponds to each row
    # having been advanced by ONE pseudo-timestep!)
    for sweep in range(max_sweeps): 
        # set the old solution        
        if sweep == 0:
            # for first sweep, set intial guess for periodic solution
            f_TS_init = [init_value for index in indices]
            #f_TS_init = [random.random()*init_value for index in indices]
            f_TS_hist.append(f_TS_init)
            f_TS_old = f_TS_init
            delta_tau = delta_tau_init        
        else:
            f_TS_old = f_TS_new
        
        # record the pseudo-timestep to be used for this sweep
        delta_tau_hist.append(delta_tau)
        # clear the new solution holder
        f_TS_new = []
        
        # perform one sweep (one pseudo-time step) down the rows       
        for i in range(N):    
            # rename/isolate the solution of the current row
            f_TS_i = f_TS_old[i]    
            # compute the source term for this row
            rows_above = 0
            for j in range(i):
                rows_above += D[i][j]*f_TS_new[j]
            rows_below = 0
            for j in range(i+1,N):
                rows_below += D[i][j]*f_TS_old[j]
            diag_source = D[i][i]*f_TS_old[i]    # (should be zero for TS)
            row_source =  rows_above + diag_source + rows_below
            
            # compute dfdt for this row from the ODE
            dfdt_i = myPeriodicODE(t[i], f_TS_old[i])            
            # calculate the residual for this pseudo-iteration
            row_residual = dfdt_i - row_source
            
            # update the solution at this time instance
            f_TS_i += delta_tau*row_residual            
            # store the updated solution for this row
            f_TS_new.append(f_TS_i)
        
        # append newly computed solution
        f_TS_hist.append(f_TS_new)
        # compute D*f
        Df = myMult(D,f_TS_new)
        # compute dfdt from the ODE
        dfdt = [myPeriodicODE(t[index],f_TS_new[index]) for index in indices]
        # compute the residual vector for this solution
        res = [-Df[index]+dfdt[index] for index in indices]
        # find the norm of the residual vector and store it
        res_hist.append(myNorm(res))
        # print the progress of the solution
        if adjust_delta_tau:
            print '[ sweep: ', sweep, ']', \
                  'residual = ', res_hist[sweep], '; delta_tau = ', delta_tau
        else:
            print '[ sweep: ', sweep, ']', \
                  'residual = ', res_hist[sweep]
        # check residual stopping condition
        if res_hist[sweep] < conv_criteria:
            print '\n\tsolution found. (', sweep+1, 'sweeps required )'
            print '\t[intial guess: constant solution at', init_value,']\n'
            break
        else:
            # adjust the pseudo-timestep
            if adjust_delta_tau == True and sweep > 1:
                current_res = res_hist[sweep]
                previous_res = res_hist[sweep-2]
                ratio = previous_res/current_res                      
                #sol_diff = abs(f_TS_i-f_TS_i_old)
                #ratio = int(res_ratio)+1.2*(res_ratio%1)
                scaling_fac = 1.0
                delta_tau_new = delta_tau*scaling_fac*ratio
                #if delta_tau_new > delta_tau_init:
                delta_tau = delta_tau_new
                
                # switch to initial value
                if ratio < 1.0:
                    delta_tau = delta_tau_init
                else:
                    delta_tau = delta_tau_new
                               
                ## limit slope 
                #delta_tau_slope = (delta_tau_new-delta_tau)/1.0   # 1 sweep
                #if delta_tau_slope > 1.0:
                #    delta_tau = delta_tau
                #else:
                #    delta_tau = delta_tau_new
               
                
    # plotting: user input! do you want to animate the solution history or just
    # plot the final result? (True = animate, False = just print final result)
    animate_plot = False                  
    n_images = sweep+1
    skip_images = n_images/15
    
    # plotting: initializations and sizing
    fig = plt.figure()
    guess_ext = ' - guess - '+str(init_value).replace('.',',')
    if adjust_delta_tau == True:
        stretch = 2.5
        plots = 3
        res_plot_title = ' '
        plot_name = 'TS G-S variable step' + guess_ext
    else:
        stretch = 2.0
        plots = 2
        res_plot_title = r'$\Delta\tau = '+str(delta_tau)+'$'
        dTau_ext = str(delta_tau_init).replace('.',',')
        plot_name = 'TS G-S constant step - ' + dTau_ext + guess_ext
    xdim, ydim = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches(stretch*xdim, ydim, forward=True)

    # plotting: things that will not be changing inside the loop
    #plt.rc('text', usetex=True)               # for using latex
    #plt.rc('font', family='serif')            # setting font
    # solution history plot
    plt.subplot(1,plots,1)
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
        y1_start = min(ave_init_value, max_final+white_space)
        y1_finish = max(ave_init_value, max_final+white_space)        
        plt.ylim(y1_start, y1_finish)
    else:
        plt.ylim(min_final-white_space,max_final+white_space)
    # residual history plot
    plt.subplot(1,plots,2)
    res, = plt.semilogy([], [],'b-',label='residual')
    plt.xlabel(r'$Gauss\textnormal{-}Seidel \, sweep$', fontsize=18)
    plt.ylabel(r'$\|R\|$', fontsize=18)
    plt.title(res_plot_title)
    plt.xlim(0,sweep)
    min_power = int(math.log(min(res_hist),10))-1
    max_power = int(math.log(max(res_hist),10))+1
    plt.ylim(pow(10,min_power), pow(10,max_power))
    # pseudo-timestep history plot
    if adjust_delta_tau == True:
        plt.subplot(1,plots,3)
        dTau, = plt.plot([],[],'r-')
        plt.xlabel(r'$Gauss\textnormal{-}Seidel \, sweep$', fontsize=18)
        plt.ylabel(r'$\Delta\tau$', fontsize=18)
        plt.xlim(0,sweep)
        min_dTau = min(delta_tau_hist)
        max_dTau = max(delta_tau_hist)
        if adjust_delta_tau == True:
            ampl_dTau = abs(max_dTau-min_dTau)
            white_space = ampl_dTau/4.0
        else:
            white_space = delta_tau_init
        plt.ylim(min_dTau-white_space,max_dTau+white_space)
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
            plt.subplot(1,plots,1)
            dots.set_data(t,f_TS_hist[n])
            t_int,f_TS_int, dummy1 = fourierInterp(t,f_TS_hist[n])
            line.set_data(t_int,f_TS_int)
            plt.title(r'$sweep \,\#$'+'$'+str(n+1)+'$') # might need adjusting
            plt.legend(loc='best')
            # plot residual            
            plt.subplot(1,plots,2)
            res.set_data(range(n),res_hist[:n])
            # plot pseudo-timestep
            if adjust_delta_tau == True:
                plt.subplot(1,plots,3)
                dTau.set_data(range(n),delta_tau_hist[:n])
            # progress monitor
            frame += 1
            percent_done = float(n)*100.0/(n_images-1)
            print 'capturing fig. '+plot_name+' (frame #'+str(frame)+'): ', \
                   round(percent_done,3),'%'
            writer.grab_frame()
        writer.grab_frame()
        
    # plotting: rescale the final frame to focus on the converged solution
    if animate_plot == True:
        plt.subplot(1,plots,1)
        plt.ylim(min_final-white_space,max_final+white_space)
    
    # plotting: save an image of the final frame
    print 'saving final image...'
    plt.savefig(plot_name, dpi=500)
    print 'figure saved: ' + plot_name
    
    # free memory used for the plot
    plt.close(fig)
    
    ###########################################################################
    # compare the GS time-spectral results against the time-accurate one ######
    ###########################################################################
    
    # set the number of comparison points to that of the finer solution
    n_comp_pts = max(len(t_period),len(t_int))
    t_comp_pts = myLinspace(0,T,n_comp_pts)
    
    # interpolate both solutions onto the finer grid (interpolating the finer
    # solution will return the same thing...redundant, but easier to code)
    t_period_comp, f_period_comp = linearInterp(t_period, f_period, t_comp_pts)
    t_int_comp, f_TS_int_comp = linearInterp(t_int, f_TS_int, t_comp_pts)
    
    # now, try shifting the extracted TA period, one index at a time (from 3
    # indices before to 3 indices after its current location), until the sum of
    # the differences between it and the time-spectral curve is the smallest
    max_shift = 10
    trial_shifts = range(-max_shift, max_shift+1)
    norm_diffs = []
    for shift in trial_shifts:
        f_period_shift = f_period_comp[shift:] + f_period_comp[:shift]
        diffs = [f_period_shift[i]-f_TS_int_comp[i] for i in range(n_comp_pts)]
        norm_diffs.append(myNorm(diffs,1))
    
    # recover the number of shifted indices that minimize the difference
    min_index = norm_diffs.index(min(norm_diffs))
    optimal_shift = trial_shifts[min_index]
    f_period_opt = f_period_comp[optimal_shift:]+f_period_comp[:optimal_shift]
    
    # plot boths isolated periods atop one another
    plt.figure()
    # plot the extracted time-accurate period
    if wrong_period:
        plt.plot(t_period,f_period,'b-',label=r'$time-accurate$')
    else:
        plt.plot(t_int_comp,f_period_opt,'b-', label=r'$time-accurate$')
    # plot the interpolated time-spectral solution
    plt.plot(t,f_TS_hist[-1],'ko')
    plt.plot(t_int_comp, f_TS_int_comp,'k--', \
             label=r'$time-spectral  \left(N='+str(N)+r'\right)$')
    plt.rc('text', usetex=True)               # for using latex
    plt.rc('font', family='serif')            # setting font
    matplotlib.rcParams['text.latex.preamble']=[r"\\usepackage{mathtools}"]
    plt.xlabel(r'$t$', fontsize=18)
    plt.ylabel(r'$f(t)$', fontsize=18)
    plt.legend(loc='best')
    plot_name = 'comparison - GS-TS vs TA'
    print 'saving image...'
    plt.savefig(plot_name, dpi=500)
    print 'figure saved: ' + plot_name + '\n'
    plt.close()


'''


    

    
# standard boilerplate to call the main() function
if __name__ == '__main__':
    main()