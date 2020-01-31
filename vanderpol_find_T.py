# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 02:47:05 2017

@author: Kedar
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
plt.ioff()
plt.close('all')
import webbrowser
from time_spectral import time_spectral_operator, fourierInterp
from nm_practice import my_nelder_mead

#-----------------------------------------------------------------------------#
class vdp_problem:
    '''
    class for defining and evaluating the homogeneous van der Pol equation
    '''
    # class constructor (accepts the actual omegas from the user) -------------
    def __init__(self, mu):
        # set the nonlinear damping coefficient
        self.mu = mu
        # set the number of time-spectral time instances to -1 at first
        self.N = -1
        # approx. period by mary cartwright's 1952 formula (np.log = ln)
        self.T_theoretical_cartwright = mu*(3-2*np.log(2)) + 2*2.338*mu**(-1/3)
        # approx. T using the poincare-lindstedt method (strogatz, ex.7.6.21)
        self.T_theoretical_poincare = 2.0*np.pi/(1-(1/16)*mu**2)
        
    # function to solve the problem time-accurately ---------------------------
    def euler_solution(self, x_0, v_0, delta_t, n_steps, make_plots=False, 
                           verbose=False):
        '''
        solve the homogeneous van der pol equation given initial position (x_0) and
        initial velocity (v_0) using the euler method. do so with user-supplied
        time step (delta_t) for a given number of steps (n_steps)
        '''
        
        # set up initial solution vector
        t_init = 0.0
        x_init = np.array([[x_0],
                           [v_0]])
        
        # print results to screen
        if verbose:
            print('\n\t\t\t*** time-accurate simulation ***\n')
            print('  iteration: 0\t' + '\t time: ' + str(round(t_init,3)) + \
            '\tposition: ' + str(np.round(x_init[0],3))[1:-1] + '\tvelocity: ' + \
            str(np.round(x_init[1],3))[1:-1])
        
        # list to store solution history
        t_hist = [0.0]
        x_hist = [x_init]
        
        # integrate solution forward in time
        for n in range(1,n_steps+1):
            # get values of x and v from the previous time step
            x_old = x_hist[-1].reshape((2,1))
            # compute the stiffness matrix
            K = np.array([[0,1],
                          [-1,-self.mu*(x_old[0]**2 - 1.0)]])
            # apply the update equation
            x_new = x_old + delta_t*np.dot(K,x_old)
            # add to the solution history
            x_hist.append(x_new)
            # record the value of the next time point (for plotting)
            t_new = t_hist[-1]+delta_t
            t_hist.append(t_new)
            # print results to screen
            if (n%100==0 or n==n_steps) and verbose:
                print('  iteration: ' + str(n) + '\t time: '+str(round(t_new,3))+\
                '\tposition: '+str(np.round(x_new[0],3))[1:-1]+'\tvelocity: '+\
                str(np.round(x_new[1],3))[1:-1])
        # extract the position and velocity histories
        position_hist = [x[0] for x in x_hist]
        velocity_hist = [x[1] for x in x_hist]
        
        if make_plots:
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
        
        # return the time histories
        return t_hist, position_hist, velocity_hist
    
    # set the number of time instances to use for time-spectral ---------------
    def set_N_time_instances(self, N):
        '''
        sets the number of time instances to use when running a time spectral
        problem
        '''
        # set the supplied value of N
        self.N = int(N)
        
    # solve the problem using time spectral -----------------------------------
    def ts_solution(self, T, x_conv_orders, max_iterations=20000, 
                    initial_guess_x=[], initial_guess_v=[], make_plots=False,
                    make_movie=False, verbose=False):
        '''
        this function will solve the homogeneous van der pol equation using the 
        time-spectral method, given a desired number of time instances and an
        assumed period. it will make plots of the converged solution, the residual
        history, and the phase portrait. should set the number of time
        instances before using.
        Input:
            - T:        assumed period of steady-state limit-cycle oscillations
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
        # make sure N has been set by the user already. otherwise, ask the user
        if self.N==-1:
            N_str = input('\nplease enter the number of time instances to '+\
            'use.\n\n\tN = ')
            self.N = int(N_str)
        
        # if T has been given in a list or numpy array, pull it out as a float
        if not type(T)==float or not type(T)==np.float64:
            T = float(T[0])
        
        # build the time-spectral operator
        D = np.array(time_spectral_operator(self.N,T))
        
        # list of time instances
        t = np.array([j*T/self.N for j in range(self.N)])
        
        # initial guess of x. assume position is a sine wave with period T
        if initial_guess_x and not initial_guess_v:
            x = np.reshape(initial_guess_x,(self.N,1))
            v = np.dot(D,x)
        elif not initial_guess_x:
            x = 2*np.sin(2.0*np.pi*t/T)*np.random.rand(self.N)
            v = -2*(2.0*np.pi/T)*np.cos(2.0*np.pi*t/T)*np.random.rand(self.N)
            np.reshape(x,(self.N,1))
            np.reshape(v,(self.N,1))
        else:
            x = np.reshape(initial_guess_x,(self.N,1))
            v = np.reshape(initial_guess_v,(self.N,1))
        
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
        r_v = -x - self.mu*(x**2-1)*v - np.dot(D,v)
        # find the norm of this residual vector
        norm_r_v = np.linalg.norm(r_v)
        # record the residual-norm history
        norm_r_v_hist = [norm_r_v]
        
        # number of predictor-corrector iterations
        n_iterations = 15000
        
        # print residual every _ iterations
        print_every = 100
        
        # method of solving Dx=v (pseudotime stepping doesn't easily converge,
        # pseudoinverse shouldn't work, b/c D is singular, but it works anyway!)
        solve_x_by = 'pseudotime'
        solve_x_by = 'pseudoinverse'
        
        # define how many orders of residual reduction constitute convergence of 
        # the x-residual and the v-residual
        x_convergence_order = x_conv_orders
        x_converged = False
        v_convergence_order = np.floor(x_convergence_order/3)
        v_converged = False
        
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
                r_v_internal = -x_old - self.mu*(x_old**2-1)*v_internal_old - np.dot(D,v_internal_old)
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
            if np.log10(norm_r_v_hist[0]/norm_r_v_hist[-1]) > v_convergence_order:
                if verbose:
                    print('\n\tv-solution convergered! (at iteration: '+str(n)+')\n')
                v_converged = True
            
            # if both x and v are converged as desired, then get out of the loop.
            # if only x is converged, then add a copy to the history (to match 
            # lengths with v) and print the next iterations number
            if x_converged and v_converged:
                x_hist.append(x_old)
                norm_r_x_hist.append(norm_r_x)
                n_iterations = n    # for plotting
                break
            elif x_converged:
                x_hist.append(x_old)
                norm_r_x_hist.append(norm_r_x)
                # print the overall iteration number
                if (n%print_every==0 or n==n_iterations) and verbose:
                    print('  iteration: '+str(n)+'\n')
                continue
            else:
                pass
            
            # solve for x
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
            
            # check convergence of x
            if np.log10(norm_r_x_hist[0]/norm_r_x_hist[-1]) > x_convergence_order:
                if verbose:
                    print('\n\tx-solution convergered! (at iteration: '+str(n)+')\n')
                x_converged = True
                
            # print the overall iteration number
            if (n%print_every==0 or n==n_iterations) and verbose:
                print('  iteration: '+str(n)+'\n')
        
        # get the time-accurate solution for the comparison phase portraits
        if make_plots or make_movie:
            # set an initial condition
            x_0 = 0.5
            v_0 = 0.0
            # pick a time step and set the number of steps
            delta_t = 0.03
            n_steps = 1400
            t_hist, position_hist, velocity_hist = self.euler_solution(x_0, v_0, 
                                                                  delta_t, n_steps)
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
        return t, x_hist[-1], v_hist[-1]
        
    # define a cost for a given T_guess using time-marching --------------------
    def step_and_check_cost(self, T_guess, verbose=False):
        '''
        this function takes in a guess for the period of steady-state limit cycle
        oscillations resulting from solving the homogeneous van der pol equation
        with a given nonlinear-damping coeffient, mu, and returns the step-&-check
        cost associated with T_guess. N is the desired number of time-spectral time
        instances.
        step-&-check cost is computed but taking a partially converged time-
        spectral solution, finding the time-instance solution (x_i, v_i) 
        corresponding to the highest absolute value of velocity (i.e. the "fastest" 
        part of the solution as it tracks around the limit cycle), and then time 
        marching from that point forward ~0.1T_guess (i.e to whichever time
        instance is closest to ~0.1T_guess) and comparing the values. if the 
        correct period has been used, then the "fast" part of the solution will 
        already be on on the limit cycle marching into the "slow" part should not
        produce any discrepancy.
        Input:
            - T_guess:  a guess for what the period might be
        Output:
            - cost:     the T-&-test cost for the given guess
            - x_ts:     the partially convergered x solution (for restart)
            - v_ts:     the partially converged v solution
        '''
        # partial level of convergence here is set in terms of the orders of 
        # magnitude reduction in x (at this point, x will be frozen and v will be
        # converged half as many orders of magnitude)
        partial_x_conv_orders = 2
        
        # max number of predictor-corrector iterations to run
        max_iter = 15000
        
        # set the number of time-accurate points that should be used to march over
        # one time-spectral time interval (i.e. to march from t_ts[n] to t_ts[n+1])
        # here, let's assume you need 30 points to get to the next time instance
        ta_points_per_ts_interval = 30
        
        # solve a time-spectral problem to this level of convergence
        t_ts,x_ts,v_ts = self.ts_solution(T=T_guess,
                                         x_conv_orders=partial_x_conv_orders,
                                         max_iterations=max_iter,
                                         verbose=False)
        
        # compute the corresponding delta_t for time marching
        delta_t = t_ts[1]/ta_points_per_ts_interval
        
        # find the time instance representing the fastest-varying part of the 
        # solution, i.e. where the absolute value of the velocity is the greatest
        max_v_index = np.argmax(abs(v_ts))    
        x_ts_max_v = x_ts[max_v_index]
        v_ts_max_v = v_ts[max_v_index]
        
        # now, roll the the arrays for velocity and position such that the first
        # entry is v_ts_max_v
        index_to_make_first = max_v_index
        rolled_v_ts = np.roll(v_ts,-index_to_make_first)
        # now, look through the first half of this rolled array and find the index
        # of the miniumum. (we only look through the first half, because otherwise
        # we might get a minimum point from the other branch of the cubic 
        # nullcline. we want the one that is closest)
        abs_v_next_half = abs(rolled_v_ts[:int(self.N/2)])
        min_v_rolled_index = np.argmin(abs_v_next_half)
        # also find out which index this corresponds to in the original t_ts array
        rolled_indices = np.roll(range(self.N),-index_to_make_first)
        min_v_index = rolled_indices[min_v_rolled_index]
        # now extract the values of x and v at this index
        x_ts_min_v = x_ts[min_v_index]
        v_ts_min_v = v_ts[min_v_index]
        # n.b. we know exactly how many time instances we have to traverse to get 
        # as close as possible to the "slow" part of the limit cycle (it is just 
        # min_v_rolled_index!)
        instances_to_step = min_v_rolled_index
        # from this we can compute the number of time-accurate steps to take
        n_ta_steps = ta_points_per_ts_interval*instances_to_step
        
        # starting from the max-|v| point, time march the solution forward as many
        # time instances as required
        t_hist, pos_hist, vel_hist = self.euler_solution(x_0=x_ts_max_v,
                                                        v_0=v_ts_max_v,
                                                        delta_t=delta_t,
                                                        n_steps=n_ta_steps)
        
        # extract the final values from this short time-accurate simulation
        x_ta_end = float(pos_hist[-1])
        v_ta_end = float(vel_hist[-1])
        
        # compute the distance between the time-accurate values and the 
        # time-spectral ones in phase space (this is the same as taking the 2-norm
        # of the differences)
        distance = np.sqrt((x_ts_min_v-x_ta_end)**2 + (v_ts_min_v-v_ta_end)**2)
        
        # define this as the "cost" we are trying to minimize
        cost = distance
        
        # print details to the screen, if desired
        if verbose:
            print('\nx_ts_max_v =',x_ts_max_v)
            print('v_ts_max_v =',v_ts_max_v)
            print('x_ts_min_v =',x_ts_min_v)
            print('v_ts_min_v =',v_ts_min_v)
            print('instances_to_step =', instances_to_step)
            print('n_ta_steps =', n_ta_steps)
            print('x_ta_end =', x_ta_end)
            print('v_ta_end =', v_ta_end)
            print('distance =', distance)
            print()
        # return the value of the cost
        return cost
    
    # find the T with the lowest cost using the nelder-mead algorithm ---------
    def ts_solution_and_find_T(self, T_guess):
        '''
        this function uses the nelder-mead simplex to find the period with the
        lowest step-&-check cost, starting from an initial guess for T. should 
        set the number of time instances before using
        '''
        # make sure N has been set by the user already. otherwise, ask the user
        if self.N==-1:
            N_str = input('\nplease enter the number of time instances to '+\
            'use.\n\n\tN = ')
            self.N = int(N_str)
        
        # put T in a list, so the nelder-mead function works correctly
        initial_guess = [T_guess]
        
        # call the nelder-mead algorithm and supply step-&-check cost function
        T_best = my_nelder_mead(self.step_and_check_cost, initial_guess, 
                        std_conv=1e-2, initial_side_length=T_guess/3, verbose=True)
        T_best = T_best.astype(float)
        
        # run a fully-converged time-solution at the best T
        t_ts, x_ts, v_ts = self.ts_solution(T=T_best, 
                                           x_conv_orders=6,
                                           make_plots=True,
                                           make_movie=True,
                                           verbose=True)
        # return the best T and the convereged solution
        return T_best, t_ts, x_ts, v_ts
#-----------------------------------------------------------------------------#
# nonlinear damping constant
# for time-spectral simulations, the value of mu will determine how many time
# instances you need! the higher the mu, the bigger the N needed
mu = 1.5

# use the class to instantiate a problem object
my_vdp_problem = vdp_problem(mu)

'''
# solve the problem time-accurately
t_hist, position_hist, velocity_hist = my_vdp_problem.euler_solution(x_0=1.0330420443910371, 
                                                                     v_0=7.9707974265302424, 
                                                                     delta_t=0.003,
                                                                     n_steps=40000, 
                                                                     make_plots=True,
                                                                     verbose=True)

# set N, guess at T, and solve the problem using the time-spectral method
my_vdp_problem.set_N_time_instances(25)
T_probable = 11.1
my_vdp_problem.ts_solution(T=T_probable, 
                           x_conv_orders=4,
                           make_plots=True,
                           make_movie=True,
                           verbose=True)

# compute the cost of a particular guess for T
T_guess = 11.1
my_vdp_problem.step_and_check_cost(T_guess, verbose=True)

'''

# find the period using nelder-mead simplices
T_guess = 11.1
N = 25
my_vdp_problem.set_N_time_instances(N)
T_best, t_ts, x_ts, v_ts = my_vdp_problem.ts_solution_and_find_T(T_guess)


