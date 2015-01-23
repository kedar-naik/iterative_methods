# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 16:48:37 2015

@author: Kedar
"""

# Solving linear systems
from matplotlib import pylab

pylab.close("all")

# The system
A = [[2, -1, 0], [-0.8, 2, -1], [0, -1, 2]]
b = [32, 71, -42]

# Using the Conjugate Gradient method
from iterative_methods import CG_method
x = CG_method(A, b)

# Using the BiConjugate method
from iterative_methods import BCG_method
x = BCG_method(A,b)