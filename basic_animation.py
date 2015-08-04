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
from matplotlib import pylab as plt          # must be called AFTER use()
from matplotlib import animation             # for specifying the writer

plt.close('all')


# domain
x = [-50.0+index*float(50-(-50))/(500-1) for index in range(500)]

# number of steps to loop over
iterations = 76

# plotting: USER INPUTS! do you want to animate the solution history or just
# plot the final result? (True = animate, False = just print final result)
animate_plot = True                     
plot_name = 'bowtie race'
n_images = iterations            # total number of images computed
skip_images = 0                # images to skip between animation frames
# for metadata
video_title = plot_name
video_description = 'trying out animation in Python!'
video_producer = 'K. R. Naik'
# could be hardcoded
fps = 15        # frames per second
dpi = 150       # dots per inch

# plotting: instantiate the figure
fig = plt.figure()

# plotting: resize the figure, N.B. forward=True modifies an existing figure
xdim, ydim = plt.gcf().get_size_inches()
plt.gcf().set_size_inches(1.2*xdim, 1.1*ydim, forward=True)

# plotting: things that will not be changing inside the loop
l, = plt.plot([], [], 'k-', label='$the \, squiggle!$')
q, = plt.plot([], [], 'go', label='$the \, points!$')
p, = plt.plot([], [], 'go')

# plotting: things that will not be changing insdie the loop
plt.rc('text', usetex=True)               # for using latex
plt.rc('font', family='serif')            # setting font
plt.xlabel(r'$x$', fontsize=18)
plt.ylabel(r'$y$', fontsize=18)
plt.xlim(-60,60)
plt.ylim(-20,20)

# plotting: set the total number of frames
if animate_plot == True:
    # capture all frames (skipping, if necessary) and the final frame
    all_frames = range(0,n_images,skip_images+1)+[n_images-1]
else:
    # no animation: just capture the last one
    all_frames = [n_images-1]

# plotting: animation preliminaries
FFMpegWriter = animation.writers['ffmpeg']
metadata = dict(title = video_title, 
                artist = video_producer,
                comment = video_description)
                
# plotting: set up the video writer
writer = FFMpegWriter(fps=fps, metadata=metadata)

# plotting: capture the video
with writer.saving(fig, plot_name+'.mp4', dpi):
    
    # frame counter
    frame = 0
    
    # loop to create and capture frames
    for i in all_frames:

        # compute the new data
        a = float(i)/25.0
        y = [(a*math.sqrt(abs(x_i)))*pow(math.sin(x_i),2.0)*math.cos(x_i) \
        for x_i in x]
        
        # plot the data
        l.set_data(x, y)
        
        # parametrized x location as an int
        fraction_traveled = math.floor(float(i)*100.0/(iterations-1))
        int_index = int(math.floor(fraction_traveled*500.0/100.0))
        q.set_data(x[int_index-1],y[int_index-1])
        p.set_data(x[-int_index],y[-int_index])
        
        # need to treat str() castings like this in order to work with Latex
        a_string= '$'+str(a)+'$'
        plt.title(r'$y = $' + a_string + r'$\sqrt{|x|}sin^2(x)cos(x)$')
        plt.legend(loc='best')
        
        # progress monitor
        percent_done = float(i)*100.0/(n_images-1)
        print 'capturing fig. '+plot_name+' (frame #'+str(frame)+'): ', \
                   percent_done,'%'
                   
        # capture the currect frame
        writer.grab_frame()
        
        # increment frame counter
        frame += 1
                
    # capture final frame outside the loop for video stability
    writer.grab_frame()
    writer.grab_frame()

# plotting: save an image of the final frame
print 'saving final image...'
plt.savefig(plot_name, dpi=1000)
print 'figure saved: ' + plot_name

# free memory used for the plot
plt.close(fig)