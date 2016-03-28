# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 18:15:35 2015

@author: Kedar
"""

from matplotlib import pylab as plt
import math

plt.close('all')
plt.rc('text', usetex=True)               # for using latex
plt.rc('font', family='serif')            # setting font

# for the 1D ODE, plot residual stall level against given (incorrect) period

# periods tried (2.0 is correct)
T = [1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]

# residual at which analysis stalls
stall_res = [3.00822747148e-11,    # T = 1.0
             3.40277508912e-11,    # T = 1.5
             3.45649060659e-11,    # T = 1.75
             3.47605031684e-11,    # T = 2.0
             3.51394994588e-11,    # T = 2.25
             3.54063857616e-11,    # T = 2.5
             3.55316119477e-11]    # T = 3.0
             
# iterations required to reach that stalled residual
iterations = [721331,    # T = 1.0
              693673,    # T = 1.5
              691735,    # T = 1.75
              684037,    # T = 2.0
              686417,    # T = 2.25
              688222,    # T = 2.5
              690531]    # T = 3.0

# plot the stalled residual values vs. the period
plot_me = False
if plot_me:
    plot_name = 'stallRes_v_T'
    plt.figure()
    plt.plot(T, stall_res, 'k.-')
    plt.plot(2.0,3.47605031684e-11,'r*')
    plt.xlabel(r'$T$', fontsize=18)
    plt.ylabel(r'$\|R\|_{stall}$', fontsize=18)
    print 'saving figure...'
    plt.savefig(plot_name, dpi=500)
    print 'figure saved: ' + plot_name
    plt.close()

# plot the iterations required to reach stalled residual vs. the period
plot_me = False
if plot_me:
    plot_name = 'iterations_v_T'
    plt.figure()
    plt.plot(T, iterations, 'k.-')
    plt.plot(2.0,684037,'r*')
    plt.xlabel(r'$T$', fontsize=18)
    plt.ylabel(r'$iterations \,\,\, required$', fontsize=18)
    print 'saving figure...'
    plt.savefig(plot_name, dpi=500)
    print 'figure saved: ' + plot_name
    plt.close()

###############################################################################
# period finding via partial convergence ######################################
###############################################################################
delta_tau = 0.0001            # pseudo-timestep used 
conv_crit = 1e1               # partial convergence level
T_actual = 2.0                # correct T
# data:   T     iterations
runs = [(0.25,   25462),
        (0.50,	25401),
        (0.75,	25376),
        (1.00,	25400),
        #(1.50,	25334),
        #(1.75,	25328),
        (2.00,	25302),
        #(2.25,	25314),
        #(2.50,	25320),
        #(2.75,	25318),
        (3.00,	25331),
        #(3.25,	25323),
        #(3.50,	25314),
        #(3.75,	25312),
        (4.00,	25302),
        #(4.25,	25302),
        #(4.50,	25309),
        #(4.75,	25309),
        (5.00,	25313),
        #(5.25,	25315),
        #(5.50,	25305),
        #(5.75,	25303),
        (6.00,	25302),	
        #(6.25,	25289),
        #(6.50,	25295),
        #(6.75,	25287),
        (7.00,	25337),
        #(7.25,  25314),
        #(7.50,	25308),
        #(7.75,	25308),
        (8.00,	25302),
        (9.00,	25298),
        (10.00,	25303),
        (11.00,	25214),
        (12.00,	25303),
        (13.00,	25312),
        (14.00,	25303),
        (15.00,	25313),
        (16.00,	25302),
        (17.00,	25302),
        (18.00,	25302),
        (19.00,	25293),
        (20.00,	25303),
        (21.00,	25294),
        (22.00,	25303),
        (23.00,	25393),
        (24.00,	25303),
        (25.00,	25308),
        (26.00,	25303),
        (48.00,  25303),
        (49.00,	25313),
        (50.00,	25303),
        (51.00,	25302),
        (74.00,	25304),
        (75.00,	25338),
        (76.00,	25304),
        (100.00,	25304)]
        
        
#-----------------------------------------------------------------------------        
 
delta_tau = 0.00005            # pseudo-timestep used 
conv_crit = 1e1               # partial convergence level
T_actual = 5.25                # correct T
# data:   T     iterations
runs = [(1.00,	50865),
        (2.00,	50755),
        (3.00,	50756),
        (4.00,	50669),
        (5.00,	50627),
        (5.25,	50607),
        (6.00,	50636),
        (7.00,	50640),
        (8.00,   50668),
        (9.00,	50631),
        (10.00,	50626),
        (11.00,	50600),
        (12.00,	50620),
        (13.00,	50623),
        (14.00,	50620),
        (15.00,	50609),
        (16.00,	50588),
        (17.00,	50601),
        (18.00,	50646)]

#-----------------------------------------------------------------------------        
 
delta_tau = 0.00005            # pseudo-timestep used 
conv_crit = 1e1               # partial convergence level
T_actual = 10.3                # correct T

# data:   T     iterations
runs = [#(1.0,	50918),
        (2.0,	50860),
        #(3.0,	50776),
        (4.0,	50757),
        #(5.0,	50757),
        (6.0,	50745),
        #(7.0,	50745),
        (8.0,	50669),
        #(9.0,	50658),
        (10.0,	50619),
        (10.30,	50608),    # actual
        #(11.0,	50609),
        (12.0,	50642),
        #(13.0,	50643),
        (14.0,	50640),
        #(15.0,	50652),
        (16.0,	50669),
        #(17.0,	50641),
        (18.0,	50630),
        #(19.0,	50628),
        (20.0,	50623),
        #(21.0,	50599),
        (22.0,	50610),
        #(23.0,	50623),
        (24.0,	50621),
        #(25.0,	50619)
]



Ts = [period for period,i in runs]
iters = [i for period,i in runs]
iter_actual = [i for period,i in runs if period == T_actual][0]

plot_me = False
if plot_me:
    plot_name = 'iter_v_T (conv='+str(int(conv_crit))+')'
    plt.figure()
    plt.plot(Ts,iters,'k.-')
    plt.plot(Ts,[iter_actual]*len(Ts),'r-')
    plt.plot(T_actual,iter_actual,'r*')
    plt.xlabel(r'$T$', fontsize=18)
    plt.ylabel(r'$iterations \,\,\, required$', fontsize=18)
    plt.title(r'$converged \,\,to \colon \,\,$'+str(conv_crit)+r'$; \,\, \Delta \tau = \,$'+str(delta_tau)+r'$; \,\, T_{actual}= \,$'+str(T_actual))
    print 'saving figure...'
    plt.savefig(plot_name, dpi=500)
    print 'figure saved: ' + plot_name
    plt.close()

######
# trends in the |error| in "extended periods"

######

# conv_crit = 1e-4 
conv_crit = 1e-4               # partial convergence level
T_actual = 2.0                # correct T

# data:   T         |error|_2
runs = [(0.50,   0.00556450468472),
        (0.75,   0.00974463806833),
        (0.90,   0.0192855495372),
        (1.00,	0.0274139041415),
        (1.10,   0.0298522384506),
        (1.25,   0.0222590428774),
        (1.50,   0.0167039461754),
        (1.75,   0.0133077715971),
        (1.90,   0.00596815914486),
        (2.00,	0.000105062780556),
        (2.10,   0.0035962361125),
        (2.25,   0.00917104019624),
        (2.50,   0.0125289156329),
        (2.75,   0.0137761667852),
        (2.90,   0.0170033383441),
        (3.00,	0.020628543089),
        (3.10,   0.0229633445253),
        (3.25,   0.0180687922596),
        (3.50,   0.0127307417351),
        (3.75,   0.0105902017216),
        (3.90,   0.0059311421671),
        (4.00,   0.0012884173493),
        (4.10,   0.00504640080272),
        (4.25,   0.00761503343852),
        (4.50,   0.0109833812199),
        (4.75,   0.0129015412852),
        (4.90,   0.0141997837238),
        (5.00,   0.0158347309533),
        (5.10,   0.0188335596592),
        (5.25,   0.0160735217446),
        (5.50,   0.00978979571466),
        (5.75,   0.00735774943002),
        (5.90,   0.00804958103873),
        (5.95,   0.00818464830201),
        (6.00,   0.00801463750565),
        (6.10,   0.0103408453224),
        (6.25,   0.0140747191932),
        (6.50,   0.0139167521764),
        (6.75,   0.0186894042911),
        (6.90,   0.0246014953198),
        (7.00,   0.0279905169866),
        (7.10,   0.0216270247645)]
y_label = r'$ \| error \|_2 $'
y_label = r'$ T{\| error \|_2}^{-3} $'


'''
# data:   T         sum(errors)
runs = [(0.50,   -0.0180891421118),
        (0.75,   -0.0245606031761),
        (0.90,   -0.0668598308119),
        (1.00,	-0.100810798789),
        (1.10,   -0.111988854045),
        (1.25,   -0.0835853819268),
        (1.50,   -0.0564458354793),
        (1.75,   -0.0416660252262),
        (1.90,   -0.0179499436191),
        (2.00,	6.49580724144e-05),
        (2.10,   0.00148073045394),
        (2.25,   -0.0193158577061),
        (2.50,   -0.0306532657669),
        (2.75,   -0.0249706887555),
        (2.90,   -0.0390949141279),
        (3.00,	-0.0591883991163),
        (3.10,   -0.0720400894074),
        (3.25,   -0.0544091238455),
        (3.50,   -0.0319668423919),
        (3.75,   -0.0250710593687),
        (3.90,   -0.0167059253284),
        (4.00,   9.41295293498e-05),
        (4.10,   0.0116079472964),
        (4.25,   0.000257634592738),
        (4.50,   -0.0172095314889),
        (4.75,   -0.0122150242641),
        (4.90,   -0.0118161897768),
        (5.00,   -0.0250625652818),
        (5.10,   -0.0452565362816),
        (5.25,   -0.0395916038729),
        (5.50,   -0.0126362773331),
        (5.75,   -0.001739184387),
        (5.90,   -0.0074627780114),
        (5.95,   -0.00696225068056),
        (6.00,   4.3770538507e-05),
        (6.10,   0.0274025155928),
        (6.25,   0.0383751179029),
        (6.50,   0.0203846168951),
        (6.75,   0.0491997173417),
        (6.90,   -0.0857400247579),
        (7.00,   -0.101579383481),
        (7.10,   -0.0613055051924)]
y_label = r'$ \Sigma error $'
'''


'''
# data:   T         |error|_1
runs = [(0.50,   0.0183780149007),
        (0.75,   0.031753174985),
        (0.90,   0.0699897019649),
        (1.00,	0.101002842152),
        (1.10,   0.111988854045),
        (1.25,   0.0835853819268),
        (1.50,   0.0593405480253),
        (1.75,   0.0463800797584),
        (1.90,   0.0204122246639),
        (2.00,	0.000336911368525),
        (2.10,   0.011884248033),
        (2.25,   0.0322653078434),
        (2.50,   0.0393816311742),
        (2.75,   0.0449419553472),
        (2.90,   0.0577970456515),
        (3.00,	0.072244301262),
        (3.10,   0.0812337209091),
        (3.25,   0.0637275693014),
        (3.50,   0.0442520412393),
        (3.75,   0.0368038570316),
        (3.90,   0.0202945194227),
        (4.00,   0.0046818942842),
        (4.10,   0.017310480805),
        (4.25,   0.0275867552042),
        (4.50,   0.038545953292),
        (4.75,   0.0447088415058),
        (4.90,   0.0414695406379),
        (5.00,   0.0536792787363),
        (5.10,   0.0636736336607),
        (5.25,   0.056183472546),
        (5.50,   0.0367299243792),
        (5.75,   0.0239246799703),
        (5.90,   0.0271748482619),
        (5.95,   0.0255915288145),
        (6.00,   0.0260965241386),
        (6.10,   0.0373146433454),
        (6.25,   0.0452379363842),
        (6.50,   0.0492149288147),
        (6.75,   0.0660580204477),
        (6.90,   0.0857400247579),
        (7.00,   0.101579383481),
        (7.10,   0.0739034250076)]
y_label = r'$ \| error \|_1 $'
'''

# parse data
Ts = [period for period,i in runs]


errors = [i for period,i in runs]
error_actual = [i for period,i in runs if period == T_actual][0]


variance = max(Ts)

errors = [i*math.exp(-pow(period,2.0)/(2.0*variance))/math.sqrt(2.0*math.pi*variance) for period,i in runs]
error_actual = [i*math.exp(-pow(period,2.0)/(2.0*variance))/math.sqrt(2.0*math.pi*variance) for period,i in runs if period == T_actual][0]


errors = [pow(period,1.0)*pow(i,1.0/pow(period,3.0)) for period,i in runs]
error_actual = [pow(period,1.0)*pow(i,1.0/pow(period,3.0)) for period,i in runs if period == T_actual][0]

# gaussian times cubed
#errors = [period*period*period*pow(i,1.0/pow(period,3.0))*math.exp(-pow(period,2.0)/(2.0*variance))/math.sqrt(2.0*math.pi*variance) for period,i in runs]
#error_actual = [period*period*period*pow(i,1.0/pow(period,3.0))*math.exp(-pow(period,2.0)/(2.0*variance))/math.sqrt(2.0*math.pi*variance) for period,i in runs if period == T_actual][0]

# plot data
plot_me = False
if plot_me:
    plot_name = 'error_v_T (conv='+str(conv_crit)+').png'
    plt.figure()
    plt.plot(Ts,errors,'k.-')
    plt.plot(T_actual,error_actual,'r*')
    plt.xlabel(r'$T$', fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    plt.title(r'$converged \,\,to \colon \, $'+str(conv_crit)+r'$; \quad T_{actual}= \,$'+str(T_actual))
    print 'saving figure...'
    plt.savefig(plot_name, dpi=500)
    print 'figure saved: ' + plot_name
    plt.close()



###########
# effective "initial condition" (function value seen at first time instance, 
# i.e. at t=0)


# data:   T         f_TS(t=0)
runs = [(0.5,    8.711),
        (1.0,    8.707),
        (1.5,    8.839),
        (2.0,    8.905),
        (2.5,    8.852),
        (3.0,    8.821),
        (3.5,    8.867),
        (4.0,    8.900),
        (5.0,    8.850)]
y_label = r'$ f_{TS}\left(t_{0}\right) $'

# parse data
Ts = [period for period,i in runs]
init_conds = [i for period,i in runs]
init_conds_actual = [i for period,i in runs if period == T_actual][0]

# plot data
plot_me = False
if plot_me:
    plot_name = 'IC_v_T.png'
    plt.figure()
    plt.plot(Ts,init_conds,'k.-')
    plt.plot(T_actual,init_conds_actual,'r*')
    plt.xlabel(r'$T$', fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    print 'saving figure...'
    plt.savefig(plot_name, dpi=500)
    print 'figure saved: ' + plot_name
    plt.close()

######
# trends in the percent difference of the time-accurate period extended from 
# the final time instance and then integrated

######

# conv_crit = 1e-4 
conv_crit = 1e-2               # partial convergence level
T_actual = 2.0                # correct T
N = 17                        # no. of time instances

# data:   T         % difference
runs = [(0.25,    ),
        (0.50,    -185.44),
        (0.75,    -127.64),
        (0.90,    -341.72),
        (1.00,	 -489.52),
        (1.05,    -581.19),
        (1.10,    -684.47),
        (1.25,    -308.85),
        (1.50,    -146.76),
        (1.75,    -92.01),
        (1.90,    -39.83),
        (1.95,    -17.27),
        #(2.00,	 -1.74),
        (2.00,    -1.08),
        (2.03,     3.67),
        #(2.05,    -5.43),
        (2.05,     5.59),
        (2.10,     2.77),
        (2.25,    -29.31),
        (2.50,    -48.02),
        (2.75,    -36.64),
        #(2.90,   0.0170033383441),
        (3.00,	 -74.02),
        #(3.10,   0.0229633445253),
        (3.25,    -84.43),
        (3.50,    -59.46),
        (3.75,    -40.28),
        #(3.90,   0.0059311421671),
        (4.00,    -6.94),
        #(4.10,   0.00504640080272),
        (4.25,     5.53),
        (4.50,    -30.52),
        (4.75,    -11.02),
        #(4.90,   0.0141997837238),
        (5.00,    -24.97),
        #(5.10,   0.0188335596592),
        (5.25,    -59.58),
        (5.50,    -2.53),
        (5.75,    -26.33),
        #(5.90,   0.00804958103873),
        #(5.95,   0.00818464830201),
        (6.00,    -6.07),
        #(6.10,   0.0103408453224),
        #(6.25,   0.0140747191932),
        (6.50,    31.17),
        #(6.75,   0.0186894042911),
        #(6.90,   0.0246014953198),
        (7.00,    -110.52),
        #(7.10,   0.0216270247645)
        ]
y_label = r'$ %_{diff} $'
y_label = r'$ T{| \%_{diff/100} |}^{-1} $'


# parse data
Ts = [period for period,i in runs]


errors = [i for period,i in runs]
error_actual = [i for period,i in runs if period == T_actual][0]


#variance = max(Ts)
#errors = [i*math.exp(-pow(period,2.0)/(2.0*variance))/math.sqrt(2.0*math.pi*variance) for period,i in runs]
#error_actual = [i*math.exp(-pow(period,2.0)/(2.0*variance))/math.sqrt(2.0*math.pi*variance) for period,i in runs if period == T_actual][0]

errors = [math.exp(pow(period,-2))*(1.0/pow(abs(i/100),1)) for period,i in runs]
error_actual = [math.exp(pow(period,-2))*(1.0/pow(abs(i/100),1)) for period,i in runs if period == T_actual][0]

# gaussian times cubed
#errors = [period*period*period*pow(i,1.0/pow(period,3.0))*math.exp(-pow(period,2.0)/(2.0*variance))/math.sqrt(2.0*math.pi*variance) for period,i in runs]
#error_actual = [period*period*period*pow(i,1.0/pow(period,3.0))*math.exp(-pow(period,2.0)/(2.0*variance))/math.sqrt(2.0*math.pi*variance) for period,i in runs if period == T_actual][0]

# plot data
plot_name = 'int_diff_v_T (conv='+str(conv_crit)+').png'
plt.figure()
plt.plot(Ts,errors,'k.-')
plt.plot(T_actual,error_actual,'r*')
plt.xlabel(r'$T$', fontsize=18)
plt.ylabel(y_label, fontsize=18)
plt.title(r'$converged \,\,to \colon \, $'+str(conv_crit)+r'$; \quad T_{actual}= \,$'+str(T_actual))
print 'saving figure...'
plt.savefig(plot_name, dpi=500)
print 'figure saved: ' + plot_name
plt.close()