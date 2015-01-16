# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 16:48:37 2015

@author: Kedar
"""

# Solving linear systems

# The system
A = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]
b = [32, 71, -42]

# Using the Conjugate Gradient method
from iterative_methods import CG_method
x = CG_method(A, b)

