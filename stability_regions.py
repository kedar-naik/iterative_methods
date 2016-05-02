# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 20:16:31 2016

@author: Kedar
"""

from matplotlib import pyplot as plt
plt.close('all')
plt.ion()

x_start = -5.0
x_end = 5.0
y_start = -5.0
y_end = 5.0

x_points = 1000
y_points = 1000

x_spacing = (x_end-x_start)/(x_points-1)
x = [x_start+i*x_spacing for i in range(x_points)]
X = y_points*[x]
y_spacing = (y_end-y_start)/(y_points-1)
Y = [[y_start+i*y_spacing]*x_points for i in range(y_points)]


Z = y_points*[x_points*[0]]
Re_parts = []
Im_parts = []
for i in range(y_points):
    for j in range(x_points):
        z = complex(X[i][j],Y[i][j])
        Z[i][j] = 1+z+pow(z,2)/2
        Z[i][j] = 1+z+pow(z,2)/2+pow(z,3)/6+pow(z,4)/24 
        if abs(Z[i][j]) <= 1.0:
            Re_parts.append(X[i][j])
            Im_parts.append(Y[i][j])


plt.figure()
plt.scatter(Re_parts,Im_parts)
plt.xlabel('Re')
plt.ylabel('Im')
plt.axis('equal')
plt.grid()
