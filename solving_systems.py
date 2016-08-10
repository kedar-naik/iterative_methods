# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 16:48:37 2015

@author: Kedar
"""
import numpy as np

# Solving linear systems
from matplotlib import pylab

pylab.close("all")

# The system
A = [[1, -1, 0], \
   [-1.5, 1, -1], \
     [0, -1, 1]]
b = [32, 71, -42]

# Using the Conjugate Gradient method
from iterative_methods import CG_method
#x = CG_method(A, b)

# Using the BiConjugate method
from iterative_methods import BCG_method
#x = BCG_method(A,b)

#--------------------------------------#
# a new system
A = np.array([[2, 1, 1, 0],
              [4, 3, 3, 1],
              [8, 7, 9, 5],
              [6, 7, 9, 8]])
b = np.array([[ 5],
              [-2],
              [ 4],
              [ 1]])
B = np.array([[  4,   3,  62,  -2],
              [ -7,   8,   3,   1],
              [-34,  -6,   3,  35],
              [  4,   6,  -1,   3]])
              
# do Gaussian Elimination
from iterative_methods import my_GE_solve
x = my_GE_solve(A,b)
print('\nx =\n '+str(x)[1:-1])
x_np = np.dot(np.linalg.inv(A),b)
print('\ninv(A)*b =\n '+str(x_np)[1:-1])

X = my_GE_solve(A,B)
print('\nX =\n '+str(X)[1:-1])
X_np = np.dot(np.linalg.inv(A),B)
print('\ninv(A)*B =\n '+str(X_np)[1:-1])
