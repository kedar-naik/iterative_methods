# -*- coding: utf-8 -*-
"""
Created on Fri Sep 04 18:01:09 2015

@author: Kedar
"""

import numpy as np
from matplotlib import pylab as plt
import math

# creating the time-spectral operator matrix
def time_spectral_operator(N,T):
    """
    This function returns the time-spectral operator matrix given a number of
    time instances and a period of oscillation.
    
    Inputs: 
      - N: number of time instances
      - T: period of oscillaton
    Output:
      - D: time-spectral operator matrix (as a numpy matrix)
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
    return np.matrix(D)
    

T = 2.0*math.pi
T=100

Ns = []
conds = []
for i in range(1,40):
    Ns.append(i)
    conds.append(np.linalg.cond(time_spectral_operator(Ns[i-1],T)))
    print 'N = ', Ns[i-1], '\t cond(D) = ', conds[i-1]
#plt.figure()
plt.plot(Ns,conds,'m.-')
plt.xlabel('N')
plt.ylabel('cond(D)')
