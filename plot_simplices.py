# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 04:04:28 2017

@author: Kedar
"""
from matplotlib import pyplot as plt
import webbrowser

initial_guess = 11.1
simplices = [(11.1, 7.4),
            (7.4, 9.25),
            (7.4, 8.325),
            (8.325, 9.25),
            (8.325, 6.475),
            (8.325, 7.4),
            (7.4, 7.8625),
            (7.4, 7.16875),
            (7.4, 7.63125)]

reflected_points        = [3.7,     5.55,   9.25,   7.4,    10.175, 6.475,  6.9375,     7.63125 ]
expansion_points        = [None,    None,   10.175, 6.475,  None,   None,   None,       7.8625  ]
inside_contractions     = [9.25,    8.325,  None,   None,   7.4,    7.8625, None,       None    ]
outside_contractions    = [None,    None,   None,   None,   None,   None,   7.16875,    None    ]

best_vertices = [simplex[0] for simplex in simplices]

plot_name = 'nelder_mead_process'
auto_open = True
plt.figure(plot_name)
plt.plot(range(len(simplices)),best_vertices,'g--', label='$best\, points$')
simplex_counter = 0
for simplex in simplices:
    plt.plot([simplex_counter]*2,simplex,'ko-')
    simplex_counter += 1
counter = 0
plotted_one = False
for point in reflected_points:
    if point and plotted_one:
        plt.plot(counter,point,'co')
    else:
        if point:
            plt.plot(counter,point,'co',label='$reflection$')
            plotted_one = True
    counter += 1
counter = 0
plotted_one = False
for point in expansion_points:
    if point and plotted_one:
        plt.plot(counter,point,'mo')
    else:
        if point:
            plt.plot(counter,point,'mo',label='$expansion$')
            plotted_one = True
    counter += 1
counter = 0
plotted_one = False
for point in inside_contractions:
    if point and plotted_one:
        plt.plot(counter,point,'yo')
    else:
        if point:
            plt.plot(counter,point,'yo',label='$inside \, contraction$')
            plotted_one = True
    counter += 1
counter = 0
plotted_one = False
for point in outside_contractions:
    if point and plotted_one:
        plt.plot(counter,point,'go')
    else:
        if point:
            plt.plot(counter,point,'go',label='$outside \, contraction$')
            plotted_one = True
    counter += 1
plt.plot(0,initial_guess,'bs')
plt.plot(len(simplices)-1,simplex[0],'rs')
plt.xlabel('$Nelder\\mathrm{-}Mead \; iteration$', fontsize=18)
plt.ylabel('$simplex \; \\left( T_i \\right)$', fontsize=18)
plt.xlim(-1,len(simplices))
#plt.legend(loc='best', fontsize=18)
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