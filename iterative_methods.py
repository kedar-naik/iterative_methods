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
    U = np.copy(A)
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
    x = np.zeros((m,1), dtype=np.complex_)
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
    y = np.zeros((m,1), dtype=np.complex_)
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
    this subroutine returns the inverse of a square, nonsingular matrix A using
    gaussian elimination and an identity matrix
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
    # check to make sure it has full rank
    rank_A = np.linalg.matrix_rank(A)
    #print('rank_A =', rank_A)
    if not rank_A == n:
        raise ValueError('\n\tSORRY...CAN\'T INVERT A SINGULAR MATRIX WITH GAUSS ' + \
              'ELIMINATION. TRY USING A PSEUDOINVERSE THAT HAS BEEN ' + \
              'DEFINED BY A THRESHOLDED SVD.\n')        
    # find the correct identity matrix
    I_m = np.eye(m)
    # solve for the solution of Ax=I
    A_inv = my_GE_solve(A,I_m)
    # return the inverse of A
    return A_inv
#-----------------------------------------------------------------------------#
def my_pinv(A, approximate_using_svd=False):
    '''
    this subroutine returns A+, the pseudoinverse of A (an m-by-n matrix).
     
    using the SVD: A = USV* --> A+ = VS+U*, where S+ is a the transpose of the 
    S matrix where the nonzero elements of S are replaced by their reciprocals.
    
    now, computing the SVD isn't cheap. if A happens to have full rank, you CAN 
    use the SVD to compute the pseudoinverse, but you can also just use the 
    Moore-Penrose pseudoinverse: A+ = inv(A*A)A*, which is identical.
    
    note that the Moore-Penrose formula can only be used when A has linearly
    independent columns, i.e. when A has full rank (which is possible without
    A being square). why? because unless you have independent columns, the 
    Gram matrix A*A (which is always square) doesn't have full rank and, 
    therefore, isn't invertible. and if you can't compute inv(A*A), then you 
    can't compute inv(A*A)A* either.
    
    Again, when you have a SQUARE A with full rank (i.e. when A is invertible), 
    A+ = inv(A). the SVD method will give you EXACTLY the same result as the 
    Moore-Penrose formula (as it does whenever you have full rank) but, because
    A is square, this result will actually be the true inverse of A. that is: 
    
       A+ = VS+U* = inv(A*A)A* = inv(A),  when A has full rank and is square.
    
    in the case where A does not have full rank, the Moore-Penrose formula is
    useless. but, we can still approximate A+ using an approximate version of
    the SVD approach. note that when A does not have full rank, there will not
    be a full set of nonzero singular values in the S matrix. since you can't 
    take the reciprocal of zero, a pseudoinverse cannot be defined. BUT, the
    pseudoinverse can be approximated by replacing S with S~, where S~ is the
    same as S except that near-zero singular have been replaced by zero. then,
    S~+ amounts to taking the reciprocals of only those singular values that
    have not been zeroed out -- no more division by zero! then, the approximate
    psuedoinverse, A~+, is given by A~+ = VS~+U*

    what counts as "near-zero"? here, i am using the same threshold used by the
    pinv() routines in MATLAB, Octave, and Numpy: t = eps*max(m,n)*s_11, where 
    eps is machine epsilon (the upper bound on relative error due to 
    floating-point rounding) and s_11 is the largest singular value (which
    should, conventionally, be in the first position along the diagonal of the
    S -- or S~ -- matrix).
    
    to avoid confusion, this function makes the user specify when to return 
    this approximate pseudoinverse by setting the argument 
    approximate_using_svd=True. in other words, it requires the user to
    know if the matrix A has full rank. 
    
    by default, the function assumes the matrix has full rank and uses the 
    Moore-Penrose formula to compute A+. if the user supplies a rank-deficient 
    matrix and fails to set the approximate_using_svd to True, an error will be 
    thrown as soon as the my_inv() function is invoked to invert A*A.
    
    i could have just used the the SVD in both cases here, but i am purposely
    trying to distinguish between the two methods and highlight the importance
    of rank in computing A+ or A~+.
    
    if you really really want to use the SVD to compute the A+ for a full-rank
    matrix, then you still can. just turn on the approximate_using_svd flag and
    know that what you get will not be an approximate pseudoinverse, but rather
    the true pseudoinverse, identical to the one given by Moore-Penrose.
    '''
    import numpy as np
    
    if approximate_using_svd: 
        # compute the truncated SVD
        U,s_truncated,V_star = my_truncated_svd(A)
        # form S+ by taking the reciprocal all positive singular values and 
        # placing them on the diagonal of a zero matrix shaped like A transpose
        S_plus_truncated = np.zeros_like(np.transpose(A))
        for i in range(len(s_truncated)):
            if s_truncated[i] > 0.0:
                S_plus_truncated[i][i] = 1.0/s_truncated[i]
        # compute the adjoint of U
        U_star = np.conjugate(np.transpose(U))
        # undo the adjoint of V
        V = np.conjugate(np.transpose(V_star))
        # compute the (approximate) pseudoinverse
        A_plus = np.dot(np.dot(V,S_plus_truncated),U_star)
        
    else:
        # use the Moore-Penrose formulae.
        #
        # from linear algebra (strang pp. 107-8): let A be an m-by-n matrix. 
        # obviously, A cannot have more than m linearly independent rows and
        # cannot have more than n linearly independent columns.
        #
        # - full row rank:
        #   when rank(A)=m and m<=n (square or more columns than rows), there 
        #   exists an infinite number of n-by-m right inverses C (such that 
        #   AC=eye(m)). so, Ax=b always has at least one solution 
        #   (x=Cb: Ax=b --> ACb=b --> Ib=b). the minimum-norm solution is 
        #   recovered by the right Moore-Penrose pseudoinverse C=A*inv(AA*)
        #        
        # - full column rank:
        #   when rank(A)=n and m>=n (square or more rows than columns), there
        #   is either one solution to Ax=b (if b happens to be in the column 
        #   space to begin with) or no solution (which is when we need to look
        #   for the least-squares solution, which is an orthogonal projection
        #   of b onto the column space of A). since we have full column rank, 
        #   the columns are linearly independent. that means A has an n-by-m
        #   left inverse B (such that BA=eye(n)) and the single solution, if
        #   it exists, is given by x=Bb. it probably doesn't exist, so we have 
        #   to settle for a least-squares solution. in either case -- the exact
        #   solution or the least-squares solution -- is given by the minimum-
        #   norm solution, which is recovered by the left Moore-Penrose 
        #   pseudoinverse B=inv(A*A)
        
        # get the dimensions of A
        (m,n) = np.shape(A)
        # check to see if A has full rank
        A_has_full_rank = my_rank(A)==min(m,n)
        # compute the appropriate pseudoinverse accordingly
        if A_has_full_rank:
            # compute the adjoint of A
            A_star = np.conjugate(np.transpose(A))            
            # if the matrix has full rank and is square (m=n), then it's
            # invertible. either pseudoinverse formula will return the true
            # inverse, inv(A). here, we'll use the left pseudoinverse formula,
            # i.e. we'll lump the square case in with the case where there are
            # more rows than columns.
            if m >= n:
                # compute the left Moore-Penrose pseudoinverse
                
                (m,n) = np.shape(np.dot(A_star,A))
                A_star_A_is_square = m==n
                A_star_A_has_full_rank = my_rank(np.dot(A_star,A))==min(m,n)
                
                #print('my_rank(np.dot(A_star,A)) =', my_rank(np.dot(A_star,A)))
                #if A_star_A_is_square and A_star_A_has_full_rank: print('\nA*A invertible!')
                
                
                A_plus = np.dot(my_inv(np.dot(A_star,A)),A_star)
            else:
                # compute the right Moore-Penrose pseudoinverse
                A_plus = np.dot(A_star,my_inv(np.dot(A,A_star)))
        else:
            # if you don't have full rank, need to approximate the 
            # pseudoinverse by using a thresholded SVD
            raise ValueError('\n\tTHE MATRIX DOES NOT HAVE FULL RANK.' + \
                             '\n\t CAN\'T USE MOORE-PENROSE FORMULAS.' + \
                             '\n\tCOMPUTE AN APPROXIMATE PSEUDOINVERSE' + \
                             '\n\tBY USING A TRUNCATED SVD.\n')
    
    # return the pseudoinverse (or approximate pseudoinverse)
    return A_plus
    
#-----------------------------------------------------------------------------#
def my_rank(A):
    '''
    returns the matrix rank of a given m-by-n matrix A. 
    methodology: take the SVD. zero-out singular values that fall below a
    certain threshold. count up the number of nonzero singular values.
    '''
    # compute the truncated SVD
    U,s_truncated,V_star = my_truncated_svd(A)
    # run through the s vector, counting up the nonzero singular values
    r = 0
    for s_value in s_truncated:
        if s_value > 0.0:
            r += 1
    # return the "numerical rank" value
    return r
#-----------------------------------------------------------------------------#
def my_truncated_svd(A):
    '''
    let A be any m-by-n matrix. this function
    computes and returns an approximate SVD, where "near-zero" singular values
    are zeroed out. if a singular value falls below some set threshold, then it
    is considered to be "near-zero." what's the threshold? MATLAB, Octave, and
    NumPy use t=eps*max(m,n)*s_11, where eps is machine epsilon a.k.a. unit
    roundoff (the smallest representable positive number such that 
    1.0 + eps != 1.0 or, to say it another way, it's the maximum error that can
    occur ) and s_11 is the 
    largest singular value. Golub & Van Loan (pp. 276) propose using either
    t=eps*||A||_inf OR t=(1x10^-p)*||A||_inf, where p, an integer, is the 
    number of decimal digits of precision to which the data used to build up A 
    have -- if you're not working with experimental data, then p is machine
    precision -- and where the infinity norm of an m-by-n matrix A is given by 
    maximum row sum of A (Trefethen and Bau pp. 21). here, we'll use the 
    following, tailor-made threshold, which is a blend of the two ideas above: 
    t=(1x10^-p)*max(m,n)*s_11. 
    from Numerical Recipes: t=s_11*(eps/2)*sqrt(m+n+1)
    (see: https://docs.scipy.org/doc/numpy-dev/reference/generated/numpy.linalg.matrix_rank.html)
    note:   
        np.finfo(float).precision = the number of decimal digits to which a 
                                    float is precise
        np.finfo(float).resolution = the decimal resolution of a float, namely:
                                    1x10^-precision, 
    '''
    import numpy as np
    # select a threshold to use ('standard', 'golub', 'blend')
    threshold_type = 'standard'
    #threshold_type = 'golub'
    threshold_type = 'recipes'
    threshold_type = 'blend'
    # compute the SVD
    U,s,V_star = np.linalg.svd(A)
    # machine epsilon (the smallest representable positive number such that 
    # 1.0 + eps != 1.0)
    machine_epsilon = np.finfo(A.dtype).eps
    # machine precision (number of decimal digits to which a float is precise)
    machine_precision = np.finfo(A.dtype).precision
    # compute the infinity norm of A (which is just the largest row sum)
    norm_inf_A = np.real(max(np.sum(A,axis=1)))
    # recover the number of rows and columns in A
    m,n = np.shape(A)
    # extract the largest singular value
    s_11 = s[0] 
    # set the desired threshold
    if threshold_type == 'standard':
        threshold = machine_epsilon*max(m,n)*s_11
    if threshold_type == 'golub':
        threshold = machine_epsilon*norm_inf_A
    if threshold_type == 'recipes':
        threshold = s_11*(machine_epsilon/2.0)*np.sqrt(m+n+1)
    if threshold_type == 'blend':
        threshold = (10**-machine_precision)*max(m,n)*s_11
    # zero out singular values that fall below the threshold
    s_truncated = np.copy(s)
    for i in range(len(s)):
        if s[i] < threshold:
            s_truncated[i] = 0.0
    
    
    #print('s =', s)
    #print('standard threshold =', machine_epsilon*max(m,n)*s_11)
    #print('golub threshold =', (10**-machine_precision)*norm_inf_A)
    #print('recipes threshold =', s_11*(machine_epsilon/2.0)*np.sqrt(m+n+1))
    #print('blend threshold =', (10**-machine_precision)*max(m,n)*s_11)
    #print('s_truncated =', s_truncated)
    
    # return the truncated version of the SVD
    return U,s_truncated,V_star
#-----------------------------------------------------------------------------#

    