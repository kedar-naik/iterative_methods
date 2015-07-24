# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 20:47:32 2015

@author: Kedar
"""

import math

# [NEEDED]
# In order to save figures or movies without having them apepear in a separate 
# window, you have to use the Agg "hardcopy" backend for rendering. 
# N.B. This only works when run from the traditional Python console! 
#      It does not work in iPython!!!
#      This is because iPython selects a "user interface" backend for you.
import matplotlib                            # import by iteslf first
matplotlib.use('Agg')                        # use Anti-Grain Geometry backend
from matplotlib import pylab as plt          # must be called AFTER use()
from matplotlib import animation             # for specifying the writer

plt.close('all')

# [NEEDED] animation preliminaries
writer = animation.writers['ffmpeg'](fps=15)

# [NEEDED]
fig = plt.figure()
l, = plt.plot([], [], 'k.-')

# things that will not be changing insdie the loop
plt.rc('text', usetex=True)               # for using latex
plt.rc('font', family='serif')            # setting font
plt.xlabel(r'$x$', fontsize=18)
plt.ylabel(r'$y$', fontsize=18)
plt.xlim(-60,60)
plt.ylim(-20,20)

# plotting domain
x = [-50.0+index*float(50-(-50))/(500-1) for index in range(500)]

# number of steps to loop over
iterations = 76

# [NEEDED] invoke the mpeg writer (dpi=100)
with writer.saving(fig, 'lean_animation.mp4', 100):
    
    # loop to create and capture frames
    for i in range(iterations):

        # progress monitor
        print 'progress: ', float(i)*100.0/(iterations-1), '%'
        
        # compute the new data
        a = float(i)/25.0
        y = [(a*math.sqrt(abs(x_i)))*pow(math.sin(x_i),2.0)*math.cos(x_i) \
        for x_i in x]
        
        # [NEEDED] plot the data
        l.set_data(x, y)
        
        # need to treat str() castings like this in order to work with Latex
        a_string= '$'+str(a)+'$'
        plt.title(r'$y = $' + a_string + r'$\sqrt{|x|}sin^2(x)cos(x)$')
        
        # [NEEDED] capture the currect frame
        writer.grab_frame()
    
    # [NEEDED] capture final frame outside the loop for video stability
    writer.grab_frame()

"""
# code for saving a simple image
import matplotlib
matplotlib.use('Agg')
import pylab as plt
plt.plot([1,2,3])
plt.savefig('myfig')
"""