#!/usr/bin/env python

# Use this script for plotting SU2 residual history
# Author: Kedar R. Naik
# Date: October 2017

import matplotlib.pyplot as plt
import webbrowser

################################# user inputs #################################

# filename
results_filename = 'history.dat'
results_filename = 'history_Ny.dat'

# select the desired x-axis variable and set the axis label
x_variable = 'iteration'
x_label = '$\\tau \;\; iteration$'

# select the dersired y-axis variable and set the axis label
# pick from: "Iteration","CLift","CDrag","CSideForce",
#            "CMx","CMy","CMz","CFx","CFy","CFz","CL/CD",
#            "Res_Flow[0]","Res_Flow[1]","Res_Flow[2]",
#            "Res_Flow[3]","Res_Flow[4]", "Linear_Solver_Iterations",
#            "CFL_Number","Time(min)"
y_variable  = 'res_flow[0]'
y_label = '$\\log_{10}\\left(\,\\overline{R_{\\rho}^{(0)}}\,\\right)$'

#y_variable = 'clift'
#y_label = '$c_L$'

# do you want to save the plot? if so, what should we call the file?
my_plotfile = 'R_v_iter_T1'
my_plotfile = 'R_v_iter_Nyquist'
#my_plotfile = 'cL_v_iter_Nyquist'

# automatically open the plot once it's plotted?
pop_up = True

###############################################################################

# open the results file
results_file = open(results_filename,'r')

# read, extract relevant data
x = []
y = []
for line in results_file:
    # grab the first token
    first_entry = line.split()[0]
    # ignore the title and zone lines
    if first_entry=='TITLE' or first_entry=='ZONE':
        pass
    # extract the variable names from the variables line
    elif first_entry=='VARIABLES':
        variables = line.replace('"','').split()[2:][0].lower().split(',')
        x_index = variables.index(x_variable.lower())
        y_index = variables.index(y_variable.lower())
    # extract the columns of interest
    else:
        data_entries = line.lstrip(' ').rstrip('\n').split(', ')
        data_entries = [float(data_entry) for data_entry in data_entries]
        data_entries[0] = int(data_entries[0])
        # record the columns you care about
        x.append(data_entries[x_index])
        y.append(data_entries[y_index])

# plot the data as directed
plot_name = my_plotfile
auto_open = pop_up
plt.figure(plot_name)
plt.plot(x, y, 'k.-')
plt.xlabel(x_label, fontsize=18)
plt.ylabel(y_label, fontsize=18)
plt.tight_layout()
# save plot and close
print('\n\t'+'saving final image...', end='')
file_name = plot_name+'.png'
plt.savefig(file_name, dpi=300)
print('figure saved: '+plot_name)
plt.close(plot_name)
# open the saved image, if desired
if auto_open:
    webbrowser.open(file_name)
    