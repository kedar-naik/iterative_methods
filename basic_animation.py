# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 23:20:45 2015

@author: Kedar
"""

"""
A simple example of an animated plot
"""

import math

# In order to save figures or movies without having them apepear in a separate 
# window, you have to use the Agg "hardcopy" backend for rendering. 
# N.B. This only works when run from the traditional Python console! 
#      It does not work in iPython!!!
#      This is because iPython selects a "user interface" backend for you.
import matplotlib                            # import by iteslf first
matplotlib.use('Agg')                        # use Anti-Grain Geometry backend
from matplotlib import pylab as plt                # must be called AFTER use()
from matplotlib import animation as manimation     # for specifying the writer

plt.close('all')

# animation user inputs
video_filename = 'basic_animation.mp4'
video_title = 'bowtie race'
video_description = 'trying out animation in Python!'
video_producer = 'K. R. Naik'
fps = 15        # frames per second
dpi = 150       # dots per inch

# animation preliminaries
FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title = video_title, 
                artist = video_producer,
                comment = video_description)
writer = FFMpegWriter(fps=fps, metadata=metadata)

fig = plt.figure()
l, = plt.plot([], [], 'k-', label='the squiggle!')
q, = plt.plot([], [], 'go', label='the points!')
p, = plt.plot([], [], 'go')

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

# invoke the mpeg writer
with writer.saving(fig, video_filename, dpi):
    
    # loop to create and capture frames
    for i in range(iterations):

        # progress monitor
        percent_done = math.floor(float(i)*100.0/(iterations-1))
        print 'progress: ', percent_done, '%'
        
        # compute the new data
        a = float(i)/25.0
        y = [(a*math.sqrt(abs(x_i)))*pow(math.sin(x_i),2.0)*math.cos(x_i) \
        for x_i in x]
        
        # plot the data
        l.set_data(x, y)
        
        # parametrized x location as 
        int_index = int(math.floor(percent_done*500.0/100.0))
        q.set_data(x[int_index-1],y[int_index-1])
        p.set_data(x[-int_index],y[-int_index])
        
        # need to treat str() castings like this in order to work with Latex
        a_string= '$'+str(a)+'$'
        plt.title(r'$y = $' + a_string + r'$\sqrt{|x|}sin^2(x)cos(x)$')
        plt.legend(loc='best')
        
        # capture the currect frame
        writer.grab_frame()
        
        
    # capture final frame outside the loop for video stability
    writer.grab_frame()
    writer.grab_frame()

# save an image of the final frame
print 'saving final image...'
plt.savefig('bowtie',dpi=1000)
print 'done saving!'