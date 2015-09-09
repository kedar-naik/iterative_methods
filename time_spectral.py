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
    #f = [pow(math.sin(2*math.pi*t_i/T),3)+pow(math.cos(2*math.pi*t_i/T),3)+9 \
     #    for t_i in t]
    #dfdt = [(-1/T)*(3*math.pi*math.sin(4*math.pi*t_i/T)* \
      #    (math.cos(2*math.pi*t_i/T) - math.sin(2*math.pi*t/T))) for t_i in t] 
  
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
    C = 10                    # carrying capacity
    h = 0.5                   # determines total rate of periodic harvesting
    b = 2*math.pi/T           # b = 2*pi/period of the sinusoidal function
    
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
        print "\n Oh no! \n The number of desired points given to " + \
        "myLinsapce() must be greater than or equal to two! \n " + \
        "Otherwise, there is a divide-by-zero error!"
    
    if len(points) == 0:
        print "\n Please check the inputs to the myLinspace function." + \
        "\n The list of points is empty! Something has gone wrong..."
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
def linearInterp(x, y, x_int=None):
    
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
    plot_name = 'period extraction process'
    print '\nsaving image...'
    plt.savefig(plot_name, dpi=500)
    print 'figure saved: ' + plot_name
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
    t_period = myLinspace(0,T,points_period)
    t_period, f_period = linearInterp(t_period_0, f_period_0, t_period)
    
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
    plot_figure = True
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
        print 'saving fig. ' + plot_name + '...'
        plt.savefig(plot_name, dpi=500)
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
    plt.savefig(plot_name, dpi=500)
    print 'figure saved: ' + plot_name
    
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
    plt.rc('text', usetex=True)               # for using latex
    plt.rc('font', family='serif')            # setting font
    matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{mathtools}"]
    plt.xlabel(r'$t$', fontsize=18)
    plt.ylabel(r'$f(t)$', fontsize=18)
    plt.title(r'$\textit{steady-state period extracted from '+\
                r'time-accurate result}$')
    plot_name = 'isolated period extracted'
    print 'saving image...'
    plt.savefig(plot_name, dpi=500)
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
        start = min(ave_init_value, max_final+white_space)
        finish = max(ave_init_value, max_final+white_space)        
        plt.ylim(start, finish)
    else:
        start = min(min_final-white_space,max_final+white_space)
        finish = max(min_final-white_space,max_final+white_space)
        plt.ylim(start, finish)
    # residual history plot
    plt.subplot(1,2,2)
    res, = plt.semilogy([], [],'b-',label='residual')
    plt.xlabel(r'$iteration$', fontsize=18)
    plt.ylabel(r'$\|R\|$', fontsize=18)
    plt.title(r'$\Delta\tau = '+str(delta_tau)+'$')
    plt.xlim(0,iteration)
    min_power = int(math.log(min(res_hist),10))-1
    max_power = int(math.log(max(res_hist),10))+1
    plt.ylim(pow(10,min_power), pow(10,max_power))
    
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
    plt.savefig(plot_name, dpi=500)
    print 'figure saved: ' + plot_name
    
    # free memory used for the plot
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
    plt.plot(t_int_comp,f_period_opt,'b-', label=r'$\textit{time-accurate}$')
    # plot the interpolated time-spectral solution
    plt.plot(t,f_TS_hist[-1],'ko')
    plt.plot(t_int_comp, f_TS_int_comp,'k--', \
             label=r'$\textit{time-spectral } \left(N='+str(N)+r'\right)$')
    plt.rc('text', usetex=True)               # for using latex
    plt.rc('font', family='serif')            # setting font
    matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{mathtools}"]
    plt.xlabel(r'$t$', fontsize=18)
    plt.ylabel(r'$f(t)$', fontsize=18)
    plt.legend(loc='best')
    plot_name = 'comparison - TS vs TA'
    print 'saving image...'
    plt.savefig(plot_name, dpi=500)
    print 'figure saved: ' + plot_name
    plt.close()
    
    ###########################################################################
    # [time spectral] Gauss-Seidel analogy w/ variable pseudo-timestep ########
    ###########################################################################    
    import random
    
    adjust_delta_tau = False      # allow pseudo-time to change during run
    delta_tau_init = 0.0927       # pseudo-timestep
    #delta_tau_init = 0.0001
    
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
            dfdt_i = myPeriodicODE(t[i], T, f_TS_old[i])            
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
        dfdt = [myPeriodicODE(t[index],T,f_TS_new[index]) for index in indices]
        # compute the residual vector for this solution
        res = [-Df[index]+dfdt[index] for index in indices]
        # find the norm of the residual vector and store it
        res_hist.append(myNorm(res))
        # print the progress of the solution
        print '[ sweep: ', sweep, ']', \
              'residual = ', res_hist[sweep], '; delta_tau = ', delta_tau
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
    animate_plot = True                  
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
    plt.rc('text', usetex=True)               # for using latex
    plt.rc('font', family='serif')            # setting font
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



    

    
# standard boilerplate to call the main() function
if __name__ == '__main__':
    main()