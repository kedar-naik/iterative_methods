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
    
# this function extracts the period of steady-state oscillations ##############
#def extractPeriod(t,f):
#    """
#    Given a time-accurate solution, this function will 
#    -interpolate the given points with a Fourier series
#    -use the derivative of to figure out how far apart in time the peaks and 
#     valleys of the periodic oscillation are
#    -produce an average (RMS?)
#    """
    
# this function interpolates a series of points with a Fourier series #########
def fourierInterp(x,y):
    
    """
    This function interpolates a given set of ordinates and abscissas with a
    Fourier series. The function returns a Fourier interpolation (of the 
    highest degree trig polynomial allowed by the Nyquist Criterion) along with 
    the corresponding new set of abscissas, which are ten times as finely 
    spaced as the original.
    
    Input:
      - abscissas, x (as a list) (leave out last, duplicate point in period)
      - ordinates, y (as a list) (again, leave out last point, if periodic)
    Output:
      - new abscissas, x_int (as a list)
      - interpolated ordinates, y_int (as a list)
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
            #check this            
            #dydx_int[i] += (j+1)*(b[j+1]*math.cos((j+1)*scaled_x_int) - \
            #            a[j+1]*math.sin((j+1)*scaled_x_int))
    
    return (x_int, y_int)
    
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
    import matplotlib                        # import by iteslf first
    matplotlib.use('Agg')                    # use Anti-Grain Geometry backend
    from matplotlib import pylab as plt      # must be called AFTER use()
    from matplotlib import animation         # for specifying the writer

    # plotting preliminaries
    plt.close('all')
    writer = animation.writers['ffmpeg'](fps=15)
    
    # user inputs
    N = 7                  # number of time instaces
    T = 2*math.pi          # period of osciallation (enter a float!)
    T=1.0
    
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
    t_int, dfdt_TS_int = fourierInterp(t, dfdt_TS)
    t_int, df2dt2_TS_int = fourierInterp(t, df2dt2_TS)
    
    # fine time grid (10 times the number of time instances) for "exact" values
    t_fine = [T*index/(10*N-1) for index in range(10*N)]
    f_fine, dfdt_fine, df2dt2_fine = myPeriodicSignal(t_fine,T)
    
    # plotting
    print 'plotting fig1...'
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
    plt.plot(t_int,dfdt_TS_int,'g--',label='$(Fourier Interp.)$')
    # plot the time-spectral second dervative and interpolation
    plt.plot(t,df2dt2_TS,'yo',label='$d^2f_{TS}/dt^2')
    plt.plot(t_int,df2dt2_TS_int,'y--',label='$(Fourier Interp.)$')
    # limits, labels, legend, title
    plt.xlabel(r'$t$', fontsize=18)
    plt.ylabel('')
    plt.legend(loc='best', ncol=2)
    plt.title(r'$N = \,$'+str(N))
    # save figure
    plt.savefig('TS_verification', dpi=1000)
    print 'fig1 saved'
    
    ###########################################################################
    # [time accurate] explicit euler ##########################################
    ###########################################################################
    delta_t = 0.1
    initial_value = 8
    t_end = 20
  
    f = []
    times = []
    time_points = int((t_end/delta_t)+1)
    
    for n in range(time_points):
        times.append(n*delta_t)
        if n == 0:
            f.append(initial_value)
        else:
            f.append(f[n-1] + delta_t*myPeriodicODE(times[n-1],T,f[n-1]))
    
    
    fig = plt.figure()
    l, = plt.plot([], [],'k-',label='f')

    # things that will not be changing insdie the loop
    plt.rc('text', usetex=True)               # for using latex
    plt.rc('font', family='serif')            # setting font
    plt.xlabel(r'$t$', fontsize=18)
    plt.ylabel(r'$f(t)$', fontsize=18)
    plt.xlim(0,20)
    plt.ylim(7.8,9.0)
    plt.title(r'$\Delta t = \,$'+str(delta_t))
    
    with writer.saving(fig, 'time_accurate.mp4', 100):
        for n in range(time_points):
            l.set_data(times[:n+1],f[:n+1])
            #plt.legend(loc='best')
            writer.grab_frame()
            # progress monitor
            print 'capturing fig2: ', float(n)*100.0/(time_points-1), '%'
        writer.grab_frame()
    
    # save an image of the final frame
    print 'saving final image...'
    plt.savefig('time_accurate', dpi=1000)
    print 'fig2 saved'
    
    ###########################################################################
    # [time spectral] explict pseudo-timestepping (dfdt -> f) #################
    ###########################################################################
    delta_tau = 0.001            # pseudo-timestep
    init_value = 5.0             # constant intital guess
    max_pseudosteps = 50000      # maximum number of pseudo-timesteps to try
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
        print iteration, ":", res_hist[iteration]
        
        # check residual stopping condition
        if res_hist[iteration] < conv_criteria:
            print "\n solution found.", iteration, " iterations required." 
            break
        
        # update solution vector
        f_TS = [f_TS[index]+delta_tau*res[index] for index in indices]
                
        # store new solution for plotting
        f_TS_hist.append(f_TS)
    
    # plot of the final solution
    plt.rc('text', usetex=True)              # for using latex
    plt.rc('font',family='serif')            # setting font
    plt.figure()
    plt.plot(t,f_TS_hist[-1],'ko')
    t_int,f_TS_int = fourierInterp(t,f_TS_hist[-1])      
    plt.plot(t_int,f_TS_int,'k--')
    plt.xlabel(r'$t$', fontsize=18)
    plt.ylabel(r'$f\left(t\right)$', fontsize=18)
    plt.title(r'iteration \#'+str(iteration)+', (started from uniform solution at '+str(init_value)+')')
    
    # plot of the residual history
    plt.figure()
    plt.semilogy(range(iteration+1),res_hist,'b-')
    plt.xlabel(r'$iteration$', fontsize=18)
    plt.ylabel(r'$\|R\|$', fontsize=18)
    plt.title(r'$\Delta\tau = \,$'+str(delta_tau)+', converged at $\|R\|$ = '+str(conv_criteria))

    
# standard boilerplate to call the main() function
if __name__ == '__main__':
    main()