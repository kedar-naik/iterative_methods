# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 22:45:14 2015

@author: Kedar


"""

# This function solves a system Ax = b using the congujate gradient method.
def CG_method(A, b):
    """
    This function solves a system Ax = b using the Conjugate Gradient method.
    The Conjugate Gradient method works best when A is symmetric and positive
    definite.
    
    Inputs: A, b

    Outputs: x, plot of convergence
    """
    
    import numpy as np
    import math
    import time
    from matplotlib import pylab
    
    # convergence criteria: 2-norm of the residual is less than
    eps = 1e-10
   
    # number of rows in b
    n = len(b)
    
    # intial guess for the solution vector, x[0]
    x = []
    x.append(np.zeros(n))
    
    # intital residual vector, r[0]
    r = []
    r.append(b - np.dot(A,x[0]))
    
    # list initializations
    rho = []                            # starts at index 0
    p = [float('nan')]                  # starts at index 1
    beta = []                           # starts at index 0
    alpha = [float('nan')]              # starts at index 1
    r_norm = [np.linalg.norm(r[0])]     # starts at index 0
    
    # print the starting residual norm to the screen
    print "||r_0|| = ", r_norm[0]    
    
    # stopping criterion (maximum iterations)
    max_iter = 100
    
    # for plotting
    pylab.close("all")
    pylab.ion()
    
    
    for i in range(1, max_iter+1):
        
        # magnitude squared of previous residual vector, rho[i-1]
        rho.append(np.dot(r[i-1],r[i-1]))
        
        # comptue the search direction, p[i]
        if i == 1:
            beta.append(float('nan'))       # for consistent indexing
            p.append(r[0])
        else:
            beta.append(rho[i-1]/rho[i-2])  # scalar, beta[i-1]
            p.append(r[i-1] + beta[i-1]*p[i-1])
        
        # define vector shorthand term q_i
        q_i = np.dot(A,p[i])
        
        # define scalar step length alpha[i]
        alpha.append(rho[i-1]/np.dot(p[i],q_i))
        
        # update the solution vector, x[i]
        x.append(x[i-1] + np.dot(alpha[i],p[i]))
        
        # update the residual vector, r[i]
        r.append(r[i-1] - np.dot(alpha[i],q_i))
        
        # compute the 2-norm of the new residual vector, r[i]
        r_norm.append(np.linalg.norm(r[i]))
        
        # compute the orders of magnitude the residual has fallen
        orders_fallen = math.log10(r_norm[0]) - math.log10(r_norm[i])

        # print the progress to the screen
        print "( iteration:", i, ") ||r|| = %.10f (%.2f orders of magnitude)" \
              % (r_norm[i], orders_fallen)
        
        # plot the convergence to the screen
        pylab.plot(range(i+1), r_norm, 'ko-')
        #ax = pylab.gca()
        #ax.set_aspect('equal')
        pylab.xlabel('iteration')
        pylab.ylabel('||r||')
        pylab.draw()
        time.sleep(.01)
    
        # check for convergence
        if r_norm[i] < eps:
            
            # print the solution to the screen
            print "Conjugate Gradient Method has converged."
            print "  -No. of iterations: ", i
            print "  -Solution: x = ", x[i]
            
            break

        else:
            
            if i == max_iter:
                print "The problem has not converged." 
                print "The maximum number of iterations has been reached."
                print "If the problem appears to be converging, consider \
                      increasing the maximum number of iterations in line 52 \
                      of iterative_methods.py"
            continue
    
    # pylab interactive mode off (this keeps the plot from closing)
    pylab.ioff()
    pylab.show()
    
    return x[i]
