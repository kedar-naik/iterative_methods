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
    print("\n\t Solution Computed Using Conjugate Gradient Method \n")
    print("||r_0|| = ", r_norm[0])   
    
    # stopping criterion (maximum iterations)
    max_iter = 100
    
    # for plotting
    pylab.ion()                         # turn on interactive mode first
    pylab.figure()
    
    for i in range(1, max_iter+1):
        
        # magnitude squared of previous residual vector, rho[i-1]
        rho.append(np.dot(r[i-1],r[i-1]))
        
        # comptue the scalar improvment this step, beta[i-1], 
        # and the vector search direction, p[i]
        if i == 1:
            beta.append(float('nan'))       # for consistent indexing
            p.append(r[0])
        else:
            beta.append(rho[i-1]/rho[i-2])
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
        print("( iteration:", i, ") ||r|| = %.10f (%.2f orders of magnitude)" \
              % (r_norm[i], orders_fallen))
        
        # plot the convergence to the screen
        pylab.semilogy(range(i+1), r_norm, 'ko-')
        #ax = pylab.gca()
        #ax.set_aspect('equal')
        pylab.rc('text', usetex=True)			# for using latex
        pylab.rc('font',family='serif')			# setting font
        pylab.xlabel('iteration')
        pylab.ylabel(r'$\|r\|$')
        pylab.title('Conjugate Gradient Method')
        pylab.draw()
        time.sleep(.01)
    
        # check for convergence
        if r_norm[i] < eps:
            
            # print the solution to the screen
            print("\n Conjugate Gradient Method has converged.")
            print("  -No. of iterations: ", i)
            print("  -Solution: x = ", x[i])
            
            break

        else:
            
            if i == max_iter:
                print("The problem has not converged.")
                print("The maximum number of iterations has been reached.")
                print("If the problem appears to be converging, consider" \
                "increasing the maximum number of iterations in line 52" \
                "of iterative_methods.py")
            continue
    
    # pylab interactive mode off (this keeps the plot from closing)
    pylab.ioff()
    pylab.show()
    
    return x[i]

# This function returns the transpose of a matrix when given a list of lists
def transpose(A):
    """
    This function returns the transpose of a given matrix A.
    
    Input: A
    
    Output: transpose of A
    """
    
    # recover the matrix dimensions
    n_rows = len(A)
    n_columns = len(A[0])
    
    # initialize space for the transpose matrix
    A_transpose = [[float('nan')]*n_rows for k in range(n_columns)]
    
    # iterate through the rows and columns
    for i in range(n_rows):
        for j in range(n_columns):
            A_transpose[j][i] = A[i][j]
    
    # return the transposed matrix
    return A_transpose
    
    
# This function solves a system Ax = b using the bicongujate gradient method.
def BCG_method(A, b):
    """
    This function solves a system Ax = b using the BiConjugate Gradient method.
    The BiConjugate Gradient method works for nonsymmetric matrices A. It does
    this by replacing the orthogonal sequence of residuals (produced during 
    the standard Conjugate Gradient method) with two mutually orthogonal 
    sequences, at the price of no longer providing a minimization. For 
    symmetric, positive definite systems the method delivers the same results 
    as the Conjugate Gradient method, but at twice the cost per iteration.
    
    
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
    
    # intital bi-orthogonal residual vectors, r[0] and r_tilde[0]
    r = []
    r_tilde = []
    r.append(b - np.dot(A,x[0]))
    r_tilde.append(r[0])                # r_tilde[0] = r[0]
    
    # list initializations
    rho = []                            # starts at index 0
    p = [float('nan')]                  # starts at index 1
    p_tilde = [float('nan')]                  # starts at index 1
    beta = []                           # starts at index 0
    alpha = [float('nan')]              # starts at index 1
    r_norm = [np.linalg.norm(r[0])]     # starts at index 0
    
    # print the starting residual norm to the screen
    print("\n\t Solution Computed Using BiConjugate Gradient Method \n")
    print("||r_0|| = ", r_norm[0])
    
    # stopping criterion (maximum iterations)
    max_iter = 100
    
    # for plotting
    pylab.ion()                         # turn on interactive mode first
    pylab.figure()
    
    for i in range(1, max_iter+1):
        
        # dot the two previous residuals vector, rho[i-1]
        rho.append(np.dot(r[i-1],r_tilde[i-1]))
        
        # make sure this dot product is not equal to zero
        if rho[i-1] == 0:
            print ("The Biconjugate Gradient method is quitting in order to" \
                  "prevent a divide-by-zero error")
            import sys
            sys.exit()
                   
        # comptue the scalar improvment this step, beta[i-1], 
        # and the vector search directions, p[i] and p_tilde[0]
        if i == 1:
            beta.append(float('nan'))       # for consistent indexing
            p.append(r[0])
            p_tilde.append(r_tilde[0])
        else:
            beta.append(rho[i-1]/rho[i-2])
            p.append(r[i-1] + beta[i-1]*p[i-1])
            p_tilde.append(r_tilde[i-1] + beta[i-1]*p_tilde[i-1])
        
        # define vector shorthand terms q_i and q_tilde_i
        q_i = np.dot(A,p[i])
        q_tilde_i = np.dot(transpose(A),p_tilde[i])
        
        # define scalar step length alpha[i]
        alpha.append(rho[i-1]/np.dot(p_tilde[i],q_i))
        
        # update the solution vector, x[i]
        x.append(x[i-1] + np.dot(alpha[i],p[i]))
        
        # update the two residual vectors, r[i] and r_tilde[i]
        r.append(r[i-1] - np.dot(alpha[i],q_i))
        r_tilde.append(r_tilde[i-1] - np.dot(alpha[i],q_tilde_i))
        
        # compute the 2-norm of the new residual vector, r[i]
        r_norm.append(np.linalg.norm(r[i]))
        
        # compute the orders of magnitude the residual has fallen
        orders_fallen = math.log10(r_norm[0]) - math.log10(r_norm[i])

        # print the progress to the screen
        print("( iteration:", i, ") ||r|| = %.10f (%.2f orders of magnitude)" \
              % (r_norm[i], orders_fallen))
        
        # plot the convergence to the screen
        pylab.semilogy(range(i+1), r_norm, 'ko-')
        #ax = pylab.gca()
        #ax.set_aspect('equal')
        pylab.rc('text', usetex=True)			# for using latex
        pylab.rc('font',family='serif')			# setting font
        pylab.xlabel('iteration')
        pylab.ylabel(r'$\|r\|$')
        pylab.title('BiConjugate Gradient Method')
        pylab.draw()
        time.sleep(.01)
    
        # check for convergence
        if r_norm[i] < eps:
            
            # print the solution to the screen
            print("\n BiConjugate Gradient Method has converged.")
            print("  -No. of iterations: ", i)
            print("  -Solution: x = ", x[i])
            
            break

        else:
            
            if i == max_iter:
                print("The problem has not converged.")
                print("The maximum number of iterations has been reached.")
                print("If the problem appears to be converging, consider" \
                "increasing the maximum number of iterations in line 209" \
                "of iterative_methods.py")
            continue
    
    # pylab interactive mode off (this keeps the plot from closing)
    pylab.ioff()
    pylab.show()
    
    return x[i]

#-----------------------------------------------------------------------------#
def my_gaussian_elimination(A, partial_pivoting=True):
    '''
    this function performs Gaussian Elimination with partial pivioting on the 
    given m x m matrix. it returns the lower- and upper-triangular 
    decomposition (L and U) along with the corresponding permutation matrix, P. 
    if partial pivoting is turned off, then only L and U are returned.
    '''
    import numpy as np
    # extact the data type of A, change to float if necessary
    A_type = A.dtype
    if A_type != 'complex_':
        A_type = 'float64'
        A = A.astype(A_type)
    # find the dimensions of the matrix
    m = np.shape(A)[0]
    n = np.shape(A)[1]
    # check to make sure it's square
    assert (n == m), '\n\n\tMATRIX MUST BE SQUARE TO PERFORM GAUSS ' + \
                     'ELIMINATION! \n'+\
                     '\n\t\t\tNo. of ROWS: \t\t m = '+str(m)+'\n' + \
                     '\t\t\tNo. of COLUMNS: \t n = '+str(n)+'\n'
    # initialize the lower- and upper-triangular matrices (L=I, U=A)
    L = np.eye(m, dtype=A_type)
    U = A
    if partial_pivoting:
        P = np.eye(m, dtype=A_type)
    # run across the k columns
    for k in range(m-1):
        if partial_pivoting:
            # find the maximum of the possible pivots
            possible_pivots = [abs(U[row_index][k]) for row_index in range(k,m)]
            # find the row index that's going to get swapped
            i = k + possible_pivots.index(max(possible_pivots))
            # interchange rows i and k in U
            row_k = np.copy(U[k][k:m])   # copy row k
            U[k][k:m] = U[i][k:m]             # paste row i in the place of row k
            U[i][k:m] = row_k            # paste row k in the place of row i
            # do the same thing for L
            row_k = np.copy(L[k][:k])     # copy row k
            L[k][:k] = L[i][:k]         # paste row i in the place of row k
            L[i][:k] = row_k              # paste row k in the place of row i
            # do the same thing for P
            row_k = np.copy(P[k])   # copy row k
            P[k] = P[i]             # paste row i in the place of row k
            P[i] = row_k            # paste row k in the place of row i
        # run down the rows below the k-th row
        for j in range(k+1,m):            
            # fill in the correct multiplier in L (pp. 148-151 in Trefethen)
            L[j][k] = U[j][k]/U[k][k]
            # subtract off from the rows below k the correct multiple of row k
            U[j][k:m] = U[j][k:m] - L[j][k]*U[k][k:m]
    # return the desired decomposition
    if partial_pivoting:
        return P, L, U
    else:
        return L, U
#-----------------------------------------------------------------------------#
def my_back_substitution(U,b):
    '''
    this subroutine implements the back substitution algorithm. it solves the 
    system Ux=b, where U is upper-triangular
    '''
    import numpy as np
    # find the dimensions of the matrix
    m = np.shape(U)[0]
    n = np.shape(U)[1]
    # check to make sure it's square
    assert (n == m), '\n\n\tMATRIX MUST BE SQUARE AND UPPER-TRIANGULAR TO ' + \
                     'PERFORM BACK SUBSTITUTION!' +\
                     '\n\t\t\tNo. of ROWS: \t\t m = '+str(m)+'\n' + \
                     '\t\t\tNo. of COLUMNS: \t n = '+str(n)+'\n'
    # initialize the solution array
    x = np.zeros((m,1))
    # start back substitution
    for i in range(m-1,-1,-1):
        x[i] = b[i]
        for j in range(i+1,m):
            x[i] -= U[i][j]*x[j]
        x[i] /= U[i][i]
    # return the solution
    return x
#-----------------------------------------------------------------------------#
def my_forward_substitution(L,b):
    '''
    this subroutine implements the forward substitution algorithm. it solves 
    the system Ly=b, where L is lower-triangular
    '''
    import numpy as np
    # find the dimensions of the matrix
    m = np.shape(L)[0]
    n = np.shape(L)[1]
    # check to make sure it's square
    assert (n == m), '\n\n\tMATRIX MUST BE SQUARE AND LOWER-TRIANGULAR TO ' + \
                     'PERFORM FORWARD SUBSTITUTION!' +\
                     '\n\t\t\tNo. of ROWS: \t\t m = '+str(m)+'\n' + \
                     '\t\t\tNo. of COLUMNS: \t n = '+str(n)+'\n'
    # initialize the solution array
    y = np.zeros((m,1))
    # start the forward solve
    for i in range(m):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[i][j]*y[j]
        y[i] /= L[i][i]
    # return the solution
    return y
#-----------------------------------------------------------------------------#
def my_GE_solve(A,b):
    '''
    this subroutine solves a system Ax=b by means of gauss elimination. first,
    the matrix A is decomposed into PA=LU. then, the system that needs to be
    solved is PAx=Pb, which is the same as LUx=Pb. let y=Ux. now the system is
    Ly=Pb. solve for y. then find x by solving Ux=y. n.b. b can be a column 
    vector or a matrix. if it's a matrix, each column is treated independently.
    '''
    import numpy as np
    # find PA=LU using Gaussian Elimination with Partial Pivoting
    P, L, U = my_gaussian_elimination(A)
    # find the number of rows and columns in b
    no_of_rows = b.shape[0]
    no_of_columns = b.shape[1]
    # solve a new system for each column in b
    for j in range(no_of_columns):
        # extract the j-th column of b
        b_j = np.array([[b[i][j]] for i in range(no_of_rows)])
        # permute b_j to get the new RHS
        Pb = np.dot(P,b_j)
        # find y by solving the lower-triangular system Ly=Pb
        y = my_forward_substitution(L,Pb)
        # find x by solving the upper-triangular system Ux=y
        x_j = my_back_substitution(U,y)
        # append this solution to the previous column
        if j == 0:
            x = x_j
        else:
            x = np.hstack((x,x_j))
    # return the solution
    return x
#-----------------------------------------------------------------------------#
def my_inv(A):
    '''
    this subroutine returns the inverse of the square, nonsingular matrix A
    using gaussian elimination and an identity matrix
    '''
    import numpy as np
    # find the dimensions of the matrix
    m = np.shape(A)[0]
    n = np.shape(A)[1]
    # check to make sure it's square
    assert (n == m), '\n\n\tMATRIX MUST BE SQUARE AND NONSINGULAR TO ' + \
                     'HAVE AN INVERSE!' +\
                     '\n\t\t\tNo. of ROWS: \t\t m = '+str(m)+'\n' + \
                     '\t\t\tNo. of COLUMNS: \t n = '+str(n)+'\n'
    # find the correct identity matrix
    I_m = np.eye(m)
    # solve for the solution of Ax=I
    A_inv = my_GE_solve(A,I_m)
    # return the inverse of A
    return A_inv
#-----------------------------------------------------------------------------#
def my_pinv(A):
    '''
    this subroutine returns A+, the pseudoinverse of A, where we compute A+ 
    using the SVD: A = USV* --> A+ = VS+U*, where S+ is a the transpose of the 
    matrix where the nonzero elements of S are replaced by their reciprocals.
    '''
    import numpy as np
    # compute the SVD
    U,s,V_star = np.linalg.svd(A)
    # machine zero
    machine_zero = np.finfo(float).eps
    # our threshold for what counts as zero
    threshold = machine_zero*1e3
    # take the reciprocal of values higher than the threshold
    s_threshold_reciprocal = np.zeros(s.shape)
    for i in range(len(s)):
        if s[i] > threshold:
            s_threshold_reciprocal[i] = 1.0/s[i]
    # form S+ (put thresholded values into a diagonal matrix and transpose it)
    S_plus = np.transpose(np.diag(s_threshold_reciprocal))
    # compute the adjoint of U
    U_star = np.conjugate(np.transpose(U))
    # undo the adjoint of V
    V = np.conjugate(np.transpose(V_star))
    # compute the pseudoinverse
    A_plus = np.dot(np.dot(V,S_plus),U_star)
    '''
    # this method of finding the pseudoinverse, where A+ = inv(A*A)A*, only 
    # works when A is of full rank
    # compute the adjoint of A
    A_star = np.conjugate(np.transpose(A))
    # compute the pseudoinverse
    A_plus = np.dot(my_inv(np.dot(A_star,A)),A_star)
    '''
    # return the pseudoinverse
    return A_plus
    
#-----------------------------------------------------------------------------#
    
    