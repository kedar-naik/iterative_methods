# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 13:53:43 2017

@author: kedarnax
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
plt.ioff()
plt.close('all')
import webbrowser
from time_spectral import time_spectral_operator, fourierInterp

# initial conditions
x_0 = 0.5   # position
v_0 = 0.0   # velocity

# nonlinear damping constant (0.5)
# for time-spectral simulations, the value of mu will determine how many time
# instances you need! the higher the mu, the bigger the N needed
mu = 1.5

# physical time step
delta_t = 0.03

# total number of time steps (200)
n_steps = 1400

# set up initial solution vector
t_init = 0.0
x_init = np.array([[x_0],
                   [v_0]])

# print results to screen
print('\n\t\t\t*** time-accurate simulation ***\n')
print('  iteration: 0\t'+'\t time: '+str(round(t_init,3))+'\tposition: '+str(np.round(x_init[0],3))[1:-1]+'\tvelocity: '+str(np.round(x_init[1],3))[1:-1])

# list to store solution history
t_hist = [0.0]
x_hist = [x_init]

# integrate solution forward in time
for n in range(1,n_steps+1):
    # get values of x and v from the previous time step
    x_old = x_hist[-1]
    # compute the stiffness matrix
    K = np.array([[0,1],
                  [-1,-mu*(x_old[0]**2 - 1.0)]])
    # apply the update equation
    x_new = x_old + delta_t*np.dot(K,x_old)
    # add to the solution history
    x_hist.append(x_new)
    # record the value of the next time point (for plotting)
    t_new = t_hist[-1]+delta_t
    t_hist.append(t_new)
    # print results to screen
    if n%100==0 or n==n_steps:
        print('  iteration: '+str(n)+'\t time: '+str(round(t_new,3))+'\tposition: '+str(np.round(x_new[0],3))[1:-1]+'\tvelocity: '+str(np.round(x_new[1],3))[1:-1])
# extract the position and velocity histories
position_hist = [x[0] for x in x_hist]
velocity_hist = [x[1] for x in x_hist]

# plot the time history
plot_name = 'time_history_vdp'
auto_open = False
plt.figure(plot_name)
plt.plot(t_hist, position_hist, 'k.-')
plt.xlabel('$t$', fontsize=18)
plt.ylabel('$x$', fontsize=18)
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
    
# plot the phase plot
plot_name = 'phase_plot_vdp'
auto_open = False
plt.figure(plot_name)
plt.plot(position_hist, velocity_hist, 'k-')
plt.xlabel('$x$', fontsize=18)
plt.ylabel('$\\dot{x}$', fontsize=18)
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

# attempt to solve for steady-state van der pol limit cycle using time-spectral
#-----------------------------------------------------------------------------#
def compute_vdp_residual(x_tilde, N, T, mu):
    '''
    computes the residual of the system of systems of first-order ODEs being
    solved simultaneously at all the time instances
    Input:
        - x_tilde:  long vector containing interleaved position and velocity
                    at the time instances
        - N:        the number of time instances being used
        - T:        the assumed period of oscillation
        - mu:       the coefficient of nonliear damping being used
    Output:
        - r_tilde:  the long residual vector
    '''
    # build the time-spectral operator
    D = np.array(time_spectral_operator(N,T))
    # use the kronecker product to extend the operator matrix so that it can
    # multiply the long solution vector. (each entry in D is replaced by a 2x2 
    # diagonal matrix)
    D_tilde = np.kron(D, np.eye(2))
    # define the elongated stiffness matrix
    K_tilde = np.zeros((2*N,2*N))
    for i in range(0,2*N,2):
        K_tilde[i,i] = 0
        K_tilde[i,i+1] = 1
        K_tilde[i+1,i] = -1
        K_tilde[i+1,i+1] = -mu*(x_tilde[i]**2 - 1.0)
    # define the residual vector
    r_tilde = np.dot(K_tilde-D_tilde, x_tilde)
    # return the long residual vector
    return r_tilde
#-----------------------------------------------------------------------------#
def implicit_pseudo_step(x_tilde_old, N, T, mu, delta_tau):
    '''
    take an implicit step
    '''
    # build the time-spectral operator
    D = np.array(time_spectral_operator(N,T))
    # use the kronecker product to extend the operator matrix so that it can
    # multiply the long solution vector. (each entry in D is replaced by a 2x2 
    # diagonal matrix)
    D_tilde = np.kron(D, np.eye(2))
    # define the stiffness matrix based on the previous solution
    K = np.array([[0,1],
                  [-1,-mu*(x_tilde[0]**2 - 1.0)]])
    # use the kronecker product to extend the stiffness matrix so that it can
    # multiply the long solution vector. (each entry in an N-by-N I matrix is 
    # replaced by the above stiffness matrix)
    K_tilde = np.kron(np.eye(N), K)
    # define the residual vector
    r_tilde = np.dot(K_tilde-D_tilde, x_tilde)
    # apply the update
    x_tilde_new = x_tilde_old + np.dot(np.linalg.inv((1/delta_tau)*np.eye(2*N)+D_tilde-K_tilde), r_tilde)
    # return the updated values
    return x_tilde_new
#-----------------------------------------------------------------------------#
# number of time instances
N = 35

# assumed period of oscillation (actual is around 7.38)
T = 7.38

# list of time instances
t = np.array([j*T/N for j in range(N)])

# initial guess of x = <pos,vel>^T. assume position is a sine wave with period
# T and that velocity is a cosine wave with period T
init_positions = np.sin(2.0*np.pi*t/T)
init_velocities = -(2.0*np.pi/T)*np.cos(2.0*np.pi*t/T)

# interleave the two vectors into one long solution column vector
x_tilde = []
for i in range(N):
    x_tilde.append(init_positions[i])
    x_tilde.append(init_velocities[i])
x_tilde = np.reshape(np.array(x_tilde), (2*N,1))
# record the solution history
x_tilde_hist = [x_tilde]

# compute the residual vector corresponding to this initial guess
r_tilde = compute_vdp_residual(x_tilde, N, T, mu)
# find the norm of this residual vector
norm_r_tilde = np.linalg.norm(r_tilde)
# record the residual-norm history
norm_r_tilde_hist = [norm_r_tilde]

# pseudotime step
delta_tau = 1e-8

# number of pseudotime steps
n_pseudo_steps = 10

# type of stepping
stepping_style = 'explicit'
#stepping_style = 'implicit'

# print heading to screen
print('\n\t\t\t*** time-spectral simulation (system) ***\n')
print('  iteration: 0\t||residual||: '+str(np.round(norm_r_tilde,13)))

# drive the residual to zero by stepping forward in pseudotime
for n in range(1,n_pseudo_steps+1):
    # get the solution from the previous pseudotime step
    x_tilde_old = x_tilde_hist[-1]
    # compute the residual vector corresponding to this solution
    r_tilde = compute_vdp_residual(x_tilde_old, N, T, mu)
    # find the norm of this residual vector
    norm_r_tilde = np.linalg.norm(r_tilde)
    # record the residual-norm history
    norm_r_tilde_hist.append(norm_r_tilde)
    # apply the update equation
    if stepping_style=='explicit':
        x_tilde_new = x_tilde_old + delta_tau*r_tilde
    if stepping_style=='implicit':
        x_tilde_new = implicit_pseudo_step(x_tilde_old, N, T, mu, delta_tau)
    # add to the solution history
    x_tilde_hist.append(x_tilde_new)
    # print results to screen
    if n%1000==0 or n==n_pseudo_steps:
        print('  iteration: '+str(n)+'\t||residual||: '+str(np.round(norm_r_tilde,13)))

# extract steady-state values at the time instances for position and velocity
position_hist_ts = []
velocity_hist_ts = []
for x_tilde in x_tilde_hist:
    positions_ts = []
    velocities_ts = []
    for i in range(2*N):
        if i%2 == 0:
            positions_ts.append(x_tilde[i])
        else:
            velocities_ts.append(x_tilde[i])
    position_hist_ts.append(positions_ts)
    velocity_hist_ts.append(velocities_ts)

# plot the residual history
plot_name = 'ts_residual_vdp'
auto_open = False
plt.figure(plot_name)
plt.plot(range(n_pseudo_steps+1), norm_r_tilde_hist, 'b.-')
plt.xlabel('$n$', fontsize=18)
plt.ylabel('$\|\\mathbf{r}^n\|_2$', fontsize=18)
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

# plot the final TS points
plot_name = 'ts_sol_vdp'
auto_open = False
plt.figure(plot_name)
plt.plot(t, position_hist_ts[-1], 'bo')
t_int, positions_int, velocities_int = fourierInterp(t, position_hist_ts[-1])
plt.plot(t_int, positions_int, 'b--')
plt.xlabel('$t$', fontsize=18)
plt.ylabel('$x$', fontsize=18)
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

# plot the phase plot
plot_name = 'ts_phase_plot_vdp'
auto_open = False
plt.figure(plot_name)
plt.plot(position_hist, velocity_hist, 'k-')
plt.plot(position_hist_ts[-1], velocity_hist_ts[-1], 'bo')
plt.plot(positions_int, velocities_int, 'b--')
plt.xlabel('$x$', fontsize=18)
plt.ylabel('$\\dot{x}$', fontsize=18)
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


#------------------------- direct time-spectal approach ----------------------#
# number of time instances
N = 15

# assumed period of oscillation (actual is around 7.38)
T = 7.38

# build the time-spectral operator
D = np.array(time_spectral_operator(N,T))

# list of time instances
t = np.array([j*T/N for j in range(N)])

# initial guess of x. assume position is a sine wave with period T 
x = np.sin(2.0*np.pi*t/T)
np.reshape(x,(N,1))
x_hist = [x]

# compute the residual vector corresponding to this initial guess
r = np.dot(D,np.dot(D,x)) + mu*(x**2+1)*np.dot(D,x) + x
# find the norm of this residual vector
norm_r = np.linalg.norm(r)
# record the residual-norm history
norm_r_hist = [norm_r]

print('\n\t\t\t*** time-spectral simulation (direct) ***\n')
print('  iteration: 0\t||residual||: '+str(np.round(norm_r,13)))

# pseudotime step
delta_tau = 1e-8

# number of pseudotime steps
n_pseudo_steps = 20

# type of stepping
stepping_style = 'explicit'
#stepping_style = 'implicit'

# drive the residual to zero by stepping forward in pseudotime
for n in range(1,n_pseudo_steps+1):
    # get the solution from the previous pseudotime step
    x_old = x_hist[-1]
    # compute the residual vector corresponding to this solution
    r = np.dot(D,np.dot(D,x_old)) + mu*(x_old**2+1)*np.dot(D,x_old) + x_old
    # find the norm of this residual vector
    norm_r = np.linalg.norm(r)
    # record the residual-norm history
    norm_r_hist.append(norm_r)
    # apply the update equation
    if stepping_style=='explicit':
        x_new = x_old + delta_tau*r
    if stepping_style=='implicit':
        drdx = np.dot(D,D) + mu*np.dot(np.diag(x_old**2-1),D) + 2*mu*(x_old-1)*np.dot(D,x_old) + np.eye(N)
        x_new = x_old + np.dot(np.linalg.inv(np.eye(N)-delta_tau*drdx),delta_tau*r)
    # add to the solution history
    x_hist.append(x_new)
    # print results to screen
    if n%1000==0 or n==n_pseudo_steps:
        print('  iteration: '+str(n)+'\t||residual||: '+str(np.round(norm_r,13)))

# plot the residual history
plot_name = 'ts_direct_residual_vdp'
auto_open = False
plt.figure(plot_name)
plt.plot(range(n_pseudo_steps+1), norm_r_hist, 'b.-')
plt.xlabel('$n$', fontsize=18)
plt.ylabel('$\|\\mathbf{r}^n\|_2$', fontsize=18)
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

# plot the final TS points
plot_name = 'ts_direct_sol_vdp'
auto_open = False
plt.figure(plot_name)
plt.plot(t, x_hist[-1], 'bo')
t_int, x_int, dx_int = fourierInterp(t, x_hist[-1])
plt.plot(t_int, x_int, 'b--')
plt.xlabel('$t$', fontsize=18)
plt.ylabel('$x$', fontsize=18)
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
    

#--------------- predictor-corrector time-spectal approach -------------------#
# number of time instances
N = 25

# assumed period of oscillation (actual is around 7.38)
T = 7.38

# build the time-spectral operator
D = np.array(time_spectral_operator(N,T))

# list of time instances
t = np.array([j*T/N for j in range(N)])

# initial guess of x. assume position is a sine wave with period T 
x = 2*np.sin(2.0*np.pi*t/T)*np.random.rand(N)
v = -2*(2.0*np.pi/T)*np.cos(2.0*np.pi*t/T)*np.random.rand(N)
np.reshape(x,(N,1))
np.reshape(v,(N,1))
x_hist = [x]
v_hist = [v]

# interleave the two vectors into one long solution column vector
x_tilde = []
for i in range(N):
    x_tilde.append(x[i])
    x_tilde.append(v[i])
x_tilde = np.reshape(np.array(x_tilde), (2*N,1))

# compute the residual vector corresponding to this initial guess
r_tilde = compute_vdp_residual(x_tilde, N, T, mu)
# find the norm of this residual vector
norm_r_tilde = np.linalg.norm(r_tilde)
# record the residual-norm history
norm_r_tilde_hist = [norm_r_tilde]

# print header and initial overall residual to the screen
print('\n\t\t*** time-spectral simulation (predictor-corrector) ***\n')
print('  iteration: 0\t||residual||: '+str(np.round(norm_r_tilde,13))+'\n')

# compute the x-residual vector corresponding to this initial guess
r_x = v - np.dot(D,x)
# find the norm of this residual vector
norm_r_x = np.linalg.norm(r_x)
# record the residual-norm history
norm_r_x_hist = [norm_r_x]

# compute the v-residual vector corresponding to this initial guess
r_v = -x - mu*(x**2-1)*v - np.dot(D,v)
# find the norm of this residual vector
norm_r_v = np.linalg.norm(r_v)
# record the residual-norm history
norm_r_v_hist = [norm_r_v]

# number of predictor-corrector iterations
n_iterations = 35000

# print residual every _ iterations
print_every = 100

solve_x_by = 'pseudotime'
solve_x_by = 'pseudoinverse'

# define how many orders of residual reduction constitute convergence of the
# x-residual
freeze_x_solution = True
x_convergence_order = 7
conv_reached = False

# drive the residual to zero by stepping forward in pseudotime
for n in range(1,n_iterations+1):
    
    # get the solution from the previous pseudotime step
    x_old = x_hist[-1]
    v_old = v_hist[-1]
    
    # solve for the new v via pseudotime stepping
    v_internal_new = v_old
    # pseudotime step
    delta_tau_v = 1e-6
    # number of pseudotime steps
    n_pseudo_steps = 550
    # print every _ internal iterations
    print_every_internal = np.ceil(n_pseudo_steps/10)
    # drive the residual to zero by stepping forward in pseudotime
    for k in range(1,n_pseudo_steps+1):
        # get the solution from the previous pseudotime step
        v_internal_old = v_internal_new
        # compute the v-residual vector corresponding to this solution
        r_v_internal = -x_old - mu*(x_old**2-1)*v_internal_old - np.dot(D,v_internal_old)
        # find the norm of this residual vector
        norm_r_v_internal = np.linalg.norm(r_v_internal)
        # apply the update equation
        v_internal_new = v_internal_old + delta_tau_v*r_v_internal
        # print results to screen
        if (n%print_every==0 or n==n_iterations) and k%print_every_internal==0:
            print('    v iteration: '+str(k)+'\t||v residual||: '+str(np.round(norm_r_v_internal,13)))
    if (n%print_every==0 or n==n_iterations):
        print()
    # set the new v
    v_new = v_internal_new
    # append to the v solution and residual histories
    v_hist.append(v_new)
    norm_r_v_hist.append(norm_r_v_internal)
    
    # check convergence of x
    if freeze_x_solution:
        if np.log10(norm_r_x_hist[0]/norm_r_x_hist[-1]) > x_convergence_order:
            conv_reached = True
    else:
        conv_reached = False
        
    # solve for x, assuming you're still supposed to
    if not conv_reached:
        if solve_x_by == 'pseudotime':
            # solve for the new x via pseudotime stepping
            x_internal_new = x_old
            # pseudotime step
            delta_tau_x = 1e-8
            # number of pseudotime steps
            n_pseudo_steps = 550
            # print every _ internal iterations
            print_every_internal = np.ceil(n_pseudo_steps/10)
            # drive the residual to zero by stepping forward in pseudotime
            for k in range(1,n_pseudo_steps+1):
                # get the solution from the previous pseudotime step
                x_internal_old = x_internal_new
                # compute the x-residual vector corresponding to this solution
                r_x_internal = v_new - np.dot(D,x_internal_old)
                # find the norm of this residual vector
                norm_r_x_internal = np.linalg.norm(r_x_internal)
                # apply the update equation
                x_internal_new = x_internal_old - delta_tau_x*r_x_internal
                # print results to screen
                if (n%print_every==0 or n==n_iterations) and k%print_every_internal==0:
                    print('    x iteration: '+str(k)+'\t||x residual||: '+str(np.round(norm_r_x_internal,13)))
            if (n%print_every==0 or n==n_iterations):
                print()
            # set the new x
            x_new = x_internal_new
            norm_r_x = norm_r_x_internal
        
        if solve_x_by == 'pseudoinverse':
            # use the pseudoinverse
            x_new = np.dot(np.linalg.pinv(D),v_new)
            # compute the x-residual vector corresponding to this solution
            r_x = v_new - np.dot(D,x_new)
            # find the norm of this residual vector
            norm_r_x = np.linalg.norm(r_x)
            # print results to screen
            if (n%print_every==0 or n==n_iterations):
                print('    x pseudoinv: \t||x residual||: '+str(np.round(norm_r_x,13)))
                print()
            '''
            # use the pseudoinverse
            x_candidate = np.dot(np.linalg.pinv(D),v_new)
            # compute the x-residual vector corresponding to this solution
            r_x_candidate = v_new - np.dot(D,x_candidate)
            # find the norm of this residual vector
            norm_r_x_candidate = np.linalg.norm(r_x_candidate)
            # accept the candidate if it has a lower residual
            if norm_r_x_candidate > norm_r_x_hist[-1]:
                x_new = x_old
                norm_r_x = norm_r_x_hist[-1]
            else:
                x_new = x_candidate
                norm_r_x = norm_r_x_candidate
            '''
        # append to the x solution and residual histories
        x_hist.append(x_new)
        norm_r_x_hist.append(norm_r_x)
    else:
        # if the x-residual has converged as desired, then keep previous 
        # values for x and norm of the residual. also turn on the x-conv
        # reached flag
        x_hist.append(x_hist[-1])
        norm_r_x_hist.append(norm_r_x_hist[-1])
        if (n%print_every==0 or n==n_iterations):
            print('    x pseudoinv: \t||x residual||: '+str(np.round(norm_r_x,13)))
            print()
            
    # interleave the two solution vectors into one long solution column vector
    x_tilde = []
    for i in range(N):
        x_tilde.append(x_hist[-1][i])
        x_tilde.append(v_hist[-1][i])
    x_tilde = np.reshape(np.array(x_tilde), (2*N,1))
    # compute the overall residual vector corresponding to this vector
    r_tilde = compute_vdp_residual(x_tilde, N, T, mu)
    # find the norm of this residual vector
    norm_r_tilde = np.linalg.norm(r_tilde)
    # record the residual-norm history
    norm_r_tilde_hist.append(norm_r_tilde)
    # print the overall residual to the screen
    if (n%print_every==0 or n==n_iterations):
        print('  iteration: '+str(n)+'\t||residual||: '+str(np.round(norm_r_tilde,13))+'\n')
        
# plot the residual history
plot_name = 'ts_pred_residual_vdp'
auto_open = True
plt.figure(plot_name)
#plt.semilogy(range(n_iterations+1), norm_r_tilde_hist, 'k.-', label='$\\tilde{\\mathbf{r}}$')
plt.semilogy(range(n_iterations+1), norm_r_x_hist, 'm.-', label='$\\mathbf{r}_x$')
plt.semilogy(range(n_iterations+1), norm_r_v_hist, 'c.-', label='$\\mathbf{r}_v$')
plt.xlabel('$n$', fontsize=18)
plt.ylabel('$\|\\mathbf{r}\|_2$', fontsize=18)
plt.legend(loc='best')
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

# plot the final TS points
plot_name = 'ts_pred_sol_vdp'
auto_open = True
plt.figure(plot_name)
plt.plot(t, x_hist[-1], 'mo', label='$x_{TS}$')
t_int, x_int, dx_int = fourierInterp(t, x_hist[-1])
plt.plot(t_int, x_int, 'm--', label='$Fourier \; interp.$')
plt.plot(t, v_hist[-1], 'co', label='$v_{TS}$')
t_int, v_int, dv_int = fourierInterp(t, v_hist[-1])
plt.plot(t_int, v_int, 'c--', label='$Fourier \; interp.$')
curves_max = max(max(x_int),max(v_int))
curves_min = min(min(x_int),min(v_int))
y_span = curves_max-curves_min
y_min = curves_min-0.25*y_span
y_max = curves_max+0.25*y_span
plt.xlabel('$t$', fontsize=18)
plt.ylabel('$x\,,\,v$', fontsize=18)
plt.xlim(0,T)
plt.ylim(y_min,y_max)
plt.legend(loc='best')
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

# plot the phase plot
plot_name = 'ts_pred_phase_plot_vdp'
auto_open = True
plt.figure(plot_name)
plt.plot(position_hist, velocity_hist, 'k-', label='$time\; accurate$')
plt.plot(x_hist[-1], v_hist[-1], 'bo', label='$time\; spectral$')
plt.plot(x_int, v_int, 'b--', label='$Fourier \; interp.$')
plt.xlabel('$x$', fontsize=18)
plt.ylabel('$\\dot{x}$', fontsize=18)
plt.legend(loc='lower right')
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

# plotting: USER INPUTS!
make_movie = True
if make_movie:
    plot_name = 'ts_vdp_movie'
    movie_filename = plot_name+'.mp4'
    auto_play = False
    n_images = len(x_hist)          # total number of images computed
    skip_images = int(n_images/70)  # images to skip between animation frames
    # instantiate the figure          (denominator is approx. no. of frames)
    fig = plt.figure(plot_name)
    # rescale the figure window to fit both subplots
    xdim, ydim = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches(2.5*xdim, ydim, forward=True)
    # things that will not be changing in the loop
    # limits for the curves plot
    max_pos = max([max(x) for x in x_hist])
    min_pos = min([min(x) for x in x_hist])
    max_vel = max([max(v) for v in v_hist])
    min_vel = min([min(v) for v in v_hist])
    max_curves = max(max_pos, max_vel)
    min_curves = min(min_pos, min_vel)
    curves_span = max_curves-min_curves
    # limits for the residual plot
    max_res = max(max(norm_r_x_hist),max(norm_r_v_hist))
    min_res = min(min(norm_r_x_hist),min(norm_r_v_hist))
    # limits for the phase portrait
    pos_span = max_pos-min_pos
    vel_span = max_vel-min_vel
    # list of all frames to be captured (skipping, if needed) plus final frame
    all_frames = [0]+list(range(0,n_images,skip_images+1))+[n_images-1]
    # print message to the screen
    print('\nmaking a movie...quiet on the set!\n')
    # plotting: capturing the movie
    writer = animation.writers['ffmpeg'](fps=int(len(all_frames)/(t[-1]-t[0])))
    with writer.saving(fig, movie_filename, 300):
        # initiate the frame counter
        frame = 0
        # cycle through the frames that need to be caputured
        for n in all_frames:
            # plot TS solutions for x and v
            plt.subplot(1,3,1)
            plt.cla()
            plt.plot(t, x_hist[n], 'mo', label='$x_{TS}$')
            t_int, x_int, dx_int = fourierInterp(t, x_hist[n])
            plt.plot(t_int, x_int, 'm--', label='$Fourier \; interp.$')
            plt.plot(t, v_hist[n], 'co', label='$v_{TS}$')
            t_int, v_int, dv_int = fourierInterp(t, v_hist[n])
            plt.plot(t_int, v_int, 'c--', label='$Fourier \; interp.$')
            plt.xlabel('$t$', fontsize=18)
            plt.ylabel('$x\,,\,v$', fontsize=18)
            plt.xlim(0,T)
            plt.ylim(min_curves-0.25*curves_span, max_curves+0.25*curves_span)
            plt.legend(loc='lower left')
            # plot the x-residual history
            plt.subplot(1,3,2)
            plt.cla()
            plt.semilogy(range(n), norm_r_x_hist[:n], 'm.-', label='$\\mathbf{r}_x$')
            plt.semilogy(range(n), norm_r_v_hist[:n], 'c.-', label='$\\mathbf{r}_v$')
            plt.xlabel('$n$', fontsize=18)
            plt.ylabel('$\|\\mathbf{r}^n\|_2$', fontsize=18)
            plt.xlim(0,n_iterations)
            plt.ylim(0.1*min_res, 10*max_res)
            plt.legend(loc='lower left')
            # plot the phase portrait
            plt.subplot(1,3,3)
            plt.cla()
            plt.plot(position_hist, velocity_hist, 'k-', label='$time \; accurate$')
            plt.plot(x_hist[n], v_hist[n], 'bo', label='$time \; spectral$')
            plt.plot(x_int, v_int, 'b--', label='$Fourier \; interp.$')
            plt.xlabel('$x$', fontsize=18)
            plt.ylabel('$\\dot{x}$', fontsize=18)
            plt.xlim(min_pos-0.2*pos_span, max_pos+0.2*pos_span)
            plt.ylim(min_vel-0.2*vel_span, max_vel+0.2*vel_span)
            plt.legend(loc='lower right')
            #fig.subplots_adjust(hspace=.5)
            plt.tight_layout()
            # progress monitor
            percent_done = float(n)*100.0/(n_images-1)
            print('\tcapturing fig. '+plot_name+' (frame #'+str(frame)+'): ', \
                   round(percent_done,2),'%')
            # capture the current frame
            writer.grab_frame()
            # increment the frame counter
            frame += 1
        # grab the last frame again
        writer.grab_frame()
    # free memory used for the plot
    plt.close(plot_name)
    # print message to the screen
    print('\n\tmovie saved: '+str(movie_filename)+'\n')
    # start playing the movie once it has been saved
    if auto_play:
        webbrowser.open(movie_filename)

#-----------------------------------------------------------------------------#
def ts_vdp_solution(N, T, mu, x_conv_orders, initial_guess_x=[], 
                    initial_guess_v=[], make_plots=False, make_movie=False, 
                    verbose=False):
    '''
    this function will solve the homogeneous van der pol equation using the 
    time-spectral method, given a desired number of time instances and an
    assumed period. it will make plots of the converged solution, the residual
    history, and the phase portrait.
    Input:
        - N:    desired number of time instances
        - T:    assumed period of steady-state limit-cycle oscillations
        - mu:   constant of nonlinear damping
        - x_conv_orders:    the number of orders of magnitude reduction in the
                            x-residual (i.e. r=v-Dx) to be seen before calling
                            the solution converged.
        - initial_guess_x:  numpy array of guesses for the N values for x at
                            the time instances
        - initial_guess_v:  numpy array of guesses for the N values for x-dot 
                            at the time instances
        - make_plots:       do you want to generate the three plots?
        - make_movie:       do you want to animate the convergence process?
        - verbose:          do you want to print out the residuals?
    Output: 
        - t:    array of the N time instances
        - x:    solution values at the N time instances
        - v:    values of x-dot at the N time instances
    '''
    # build the time-spectral operator
    D = np.array(time_spectral_operator(N,T))
    
    # list of time instances
    t = np.array([j*T/N for j in range(N)])
        
    # initial guess of x. assume position is a sine wave with period T
    if initial_guess_x and not initial_guess_v:
        x = np.reshape(initial_guess_x,(N,1))
        v = np.dot(D,x)
    elif not initial_guess_x:
        x = 2*np.sin(2.0*np.pi*t/T)*np.random.rand(N)
        v = -2*(2.0*np.pi/T)*np.cos(2.0*np.pi*t/T)*np.random.rand(N)
        np.reshape(x,(N,1))
        np.reshape(v,(N,1))
    else:
        x = np.reshape(initial_guess_x,(N,1))
        v = np.reshape(initial_guess_v,(N,1))
    
    # initialize history lists
    x_hist = [x]
    v_hist = [v]
    
    # print header and initial overall residual to the screen
    if verbose:
        print('\n\t\t*** time-spectral simulation (predictor-corrector) ***\n')
        print('  iteration: 0\n')

    # compute the x-residual vector corresponding to this initial guess
    r_x = v - np.dot(D,x)
    # find the norm of this residual vector
    norm_r_x = np.linalg.norm(r_x)
    # record the residual-norm history
    norm_r_x_hist = [norm_r_x]
    
    # compute the v-residual vector corresponding to this initial guess
    r_v = -x - mu*(x**2-1)*v - np.dot(D,v)
    # find the norm of this residual vector
    norm_r_v = np.linalg.norm(r_v)
    # record the residual-norm history
    norm_r_v_hist = [norm_r_v]
    
    # number of predictor-corrector iterations
    n_iterations = 35000
    
    # print residual every _ iterations
    print_every = 100
    
    # method of solving Dx=v (pseudotime stepping doesn't easily converge,
    # pseudoinverse shouldn't work, b/c D is singular, but it works anyway!)
    solve_x_by = 'pseudotime'
    solve_x_by = 'pseudoinverse'
    
    # define how many orders of residual reduction constitute convergence of the
    # x-residual
    freeze_x_solution = True
    x_convergence_order = x_conv_orders
    conv_reached = False
    
    # drive the residual to zero by stepping forward in pseudotime
    for n in range(1,n_iterations+1):
        
        # get the solution from the previous pseudotime step
        x_old = x_hist[-1]
        v_old = v_hist[-1]
        
        # solve for the new v via pseudotime stepping
        v_internal_new = v_old
        # pseudotime step
        delta_tau_v = 1e-6
        # number of pseudotime steps
        n_pseudo_steps = 550
        # print every _ internal iterations
        print_every_internal = np.ceil(n_pseudo_steps/10)
        # drive the residual to zero by stepping forward in pseudotime
        for k in range(1,n_pseudo_steps+1):
            # get the solution from the previous pseudotime step
            v_internal_old = v_internal_new
            # compute the v-residual vector corresponding to this solution
            r_v_internal = -x_old - mu*(x_old**2-1)*v_internal_old - np.dot(D,v_internal_old)
            # find the norm of this residual vector
            norm_r_v_internal = np.linalg.norm(r_v_internal)
            # apply the update equation
            v_internal_new = v_internal_old + delta_tau_v*r_v_internal
            # print results to screen
            if (n%print_every==0 or n==n_iterations) and k%print_every_internal==0 and verbose:
                print('    v iteration: '+str(k)+'\t||v residual||: '+str(np.round(norm_r_v_internal,13)))
        if (n%print_every==0 or n==n_iterations) and verbose:
            print()
        # set the new v
        v_new = v_internal_new
        # append to the v solution and residual histories
        v_hist.append(v_new)
        norm_r_v_hist.append(norm_r_v_internal)
        
        # check convergence of x
        if freeze_x_solution:
            if np.log10(norm_r_x_hist[0]/norm_r_x_hist[-1]) > x_convergence_order:
                conv_reached = True
        else:
            conv_reached = False
            
        # solve for x, assuming you're still supposed to
        if not conv_reached:
            if solve_x_by == 'pseudotime':
                # solve for the new x via pseudotime stepping
                x_internal_new = x_old
                # pseudotime step
                delta_tau_x = 1e-8
                # number of pseudotime steps
                n_pseudo_steps = 550
                # print every _ internal iterations
                print_every_internal = np.ceil(n_pseudo_steps/10)
                # drive the residual to zero by stepping forward in pseudotime
                for k in range(1,n_pseudo_steps+1):
                    # get the solution from the previous pseudotime step
                    x_internal_old = x_internal_new
                    # compute the x-residual vector corresponding to this solution
                    r_x_internal = v_new - np.dot(D,x_internal_old)
                    # find the norm of this residual vector
                    norm_r_x_internal = np.linalg.norm(r_x_internal)
                    # apply the update equation
                    x_internal_new = x_internal_old - delta_tau_x*r_x_internal
                    # print results to screen
                    if (n%print_every==0 or n==n_iterations) and k%print_every_internal==0 and verbose:
                        print('    x iteration: '+str(k)+'\t||x residual||: '+str(np.round(norm_r_x_internal,13)))
                if (n%print_every==0 or n==n_iterations) and verbose:
                    print()
                # set the new x
                x_new = x_internal_new
                norm_r_x = norm_r_x_internal
            
            if solve_x_by == 'pseudoinverse':
                # use the pseudoinverse
                x_new = np.dot(np.linalg.pinv(D),v_new)
                # compute the x-residual vector corresponding to this solution
                r_x = v_new - np.dot(D,x_new)
                # find the norm of this residual vector
                norm_r_x = np.linalg.norm(r_x)
                # print results to screen
                if (n%print_every==0 or n==n_iterations) and verbose:
                    print('    x pseudoinv: \t||x residual||: '+str(np.round(norm_r_x,13)))
                    print()
            # append to the x solution and residual histories
            x_hist.append(x_new)
            norm_r_x_hist.append(norm_r_x)
        else:
            # if the x-residual has converged as desired, then keep previous 
            # values for x and norm of the residual. also turn on the x-conv
            # reached flag
            x_hist.append(x_hist[-1])
            norm_r_x_hist.append(norm_r_x_hist[-1])
            if (n%print_every==0 or n==n_iterations) and verbose:
                print('    x pseudoinv: \t||x residual||: '+str(np.round(norm_r_x,13)))
                print()
                
        # print the overall iteration number
        if (n%print_every==0 or n==n_iterations) and verbose:
            print('  iteration: '+str(n)+'\n')
    
    if make_plots:
        # plot the residual history
        plot_name = 'ts_pred_residual_vdp'
        auto_open = True
        plt.figure(plot_name)
        #plt.semilogy(range(n_iterations+1), norm_r_tilde_hist, 'k.-', label='$\\tilde{\\mathbf{r}}$')
        plt.semilogy(range(n_iterations+1), norm_r_x_hist, 'm.-', label='$\\mathbf{r}_x$')
        plt.semilogy(range(n_iterations+1), norm_r_v_hist, 'c.-', label='$\\mathbf{r}_v$')
        plt.xlabel('$n$', fontsize=18)
        plt.ylabel('$\|\\mathbf{r}\|_2$', fontsize=18)
        plt.legend(loc='best')
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
        
        # plot the final TS points
        plot_name = 'ts_pred_sol_vdp'
        auto_open = True
        plt.figure(plot_name)
        plt.plot(t, x_hist[-1], 'mo', label='$x_{TS}$')
        t_int, x_int, dx_int = fourierInterp(t, x_hist[-1])
        plt.plot(t_int, x_int, 'm--', label='$Fourier \; interp.$')
        plt.plot(t, v_hist[-1], 'co', label='$v_{TS}$')
        t_int, v_int, dv_int = fourierInterp(t, v_hist[-1])
        plt.plot(t_int, v_int, 'c--', label='$Fourier \; interp.$')
        curves_max = max(max(x_int),max(v_int))
        curves_min = min(min(x_int),min(v_int))
        y_span = curves_max-curves_min
        y_min = curves_min-0.25*y_span
        y_max = curves_max+0.25*y_span
        plt.xlabel('$t$', fontsize=18)
        plt.ylabel('$x\,,\,v$', fontsize=18)
        plt.xlim(0,T)
        plt.ylim(y_min,y_max)
        plt.legend(loc='best')
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
        
        # plot the phase plot
        plot_name = 'ts_pred_phase_plot_vdp'
        auto_open = True
        plt.figure(plot_name)
        plt.plot(position_hist, velocity_hist, 'k-', label='$time\; accurate$')
        plt.plot(x_hist[-1], v_hist[-1], 'bo', label='$time\; spectral$')
        plt.plot(x_int, v_int, 'b--', label='$Fourier \; interp.$')
        plt.xlabel('$x$', fontsize=18)
        plt.ylabel('$\\dot{x}$', fontsize=18)
        plt.legend(loc='lower right')
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
    
    # animation
    if make_movie:
        plot_name = 'ts_vdp_movie'
        movie_filename = plot_name+'.mp4'
        auto_play = False
        n_images = len(x_hist)          # total number of images computed
        skip_images = int(n_images/70)  # images to skip between animation frames
        # instantiate the figure          (denominator is approx. no. of frames)
        fig = plt.figure(plot_name)
        # rescale the figure window to fit both subplots
        xdim, ydim = plt.gcf().get_size_inches()
        plt.gcf().set_size_inches(2.5*xdim, ydim, forward=True)
        # things that will not be changing in the loop
        # limits for the curves plot
        max_pos = max([max(x) for x in x_hist])
        min_pos = min([min(x) for x in x_hist])
        max_vel = max([max(v) for v in v_hist])
        min_vel = min([min(v) for v in v_hist])
        max_curves = max(max_pos, max_vel)
        min_curves = min(min_pos, min_vel)
        curves_span = max_curves-min_curves
        # limits for the residual plot
        max_res = max(max(norm_r_x_hist),max(norm_r_v_hist))
        min_res = min(min(norm_r_x_hist),min(norm_r_v_hist))
        # limits for the phase portrait
        pos_span = max_pos-min_pos
        vel_span = max_vel-min_vel
        # list of all frames to be captured (skipping, if needed) plus final frame
        all_frames = [0]+list(range(0,n_images,skip_images+1))+[n_images-1]
        # print message to the screen
        print('\nmaking a movie...quiet on the set!\n')
        # plotting: capturing the movie
        writer = animation.writers['ffmpeg'](fps=int(len(all_frames)/(t[-1]-t[0])))
        with writer.saving(fig, movie_filename, 300):
            # initiate the frame counter
            frame = 0
            # cycle through the frames that need to be caputured
            for n in all_frames:
                # plot TS solutions for x and v
                plt.subplot(1,3,1)
                plt.cla()
                plt.plot(t, x_hist[n], 'mo', label='$x_{TS}$')
                t_int, x_int, dx_int = fourierInterp(t, x_hist[n])
                plt.plot(t_int, x_int, 'm--', label='$Fourier \; interp.$')
                plt.plot(t, v_hist[n], 'co', label='$v_{TS}$')
                t_int, v_int, dv_int = fourierInterp(t, v_hist[n])
                plt.plot(t_int, v_int, 'c--', label='$Fourier \; interp.$')
                plt.xlabel('$t$', fontsize=18)
                plt.ylabel('$x\,,\,v$', fontsize=18)
                plt.xlim(0,T)
                plt.ylim(min_curves-0.25*curves_span, max_curves+0.25*curves_span)
                plt.legend(loc='lower left')
                # plot the x-residual history
                plt.subplot(1,3,2)
                plt.cla()
                plt.semilogy(range(n), norm_r_x_hist[:n], 'm.-', label='$\\mathbf{r}_x$')
                plt.semilogy(range(n), norm_r_v_hist[:n], 'c.-', label='$\\mathbf{r}_v$')
                plt.xlabel('$n$', fontsize=18)
                plt.ylabel('$\|\\mathbf{r}^n\|_2$', fontsize=18)
                plt.xlim(0,n_iterations)
                plt.ylim(0.1*min_res, 10*max_res)
                plt.legend(loc='lower left')
                # plot the phase portrait
                plt.subplot(1,3,3)
                plt.cla()
                plt.plot(position_hist, velocity_hist, 'k-', label='$time \; accurate$')
                plt.plot(x_hist[n], v_hist[n], 'bo', label='$time \; spectral$')
                plt.plot(x_int, v_int, 'b--', label='$Fourier \; interp.$')
                plt.xlabel('$x$', fontsize=18)
                plt.ylabel('$\\dot{x}$', fontsize=18)
                plt.xlim(min_pos-0.2*pos_span, max_pos+0.2*pos_span)
                plt.ylim(min_vel-0.2*vel_span, max_vel+0.2*vel_span)
                plt.legend(loc='lower right')
                #fig.subplots_adjust(hspace=.5)
                plt.tight_layout()
                # progress monitor
                percent_done = float(n)*100.0/(n_images-1)
                print('\tcapturing fig. '+plot_name+' (frame #'+str(frame)+'): ', \
                       round(percent_done,2),'%')
                # capture the current frame
                writer.grab_frame()
                # increment the frame counter
                frame += 1
            # grab the last frame again
            writer.grab_frame()
        # free memory used for the plot
        plt.close(plot_name)
        # print message to the screen
        print('\n\tmovie saved: '+str(movie_filename)+'\n')
        # start playing the movie once it has been saved
        if auto_play:
            webbrowser.open(movie_filename)

    # return the desired arrays
    return t, x, v
#-----------------------------------------------------------------------------#
    
    
    
    