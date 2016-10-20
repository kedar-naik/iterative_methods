# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 19:24:44 2016

@author: Kedar
"""
import numpy as np
from matplotlib import pyplot as plt
#-----------------------------------------------------------------------------#
def whittaker_shannon_interp(t, f, t_int=False):
    '''
    this subroutine implements the Whittaker-Shannon interpolation formula.
    Input:
      - abscissas, t (as a list) (leave out last, duplicate point in period)
      - ordinates, f (as a list) (again, leave out last point, if periodic)
      - new abscissas, f_int (as a list) (optional! defaults to 10x refinement)
    Output:
      - new abscissas, t_int (np array)
      - interpolated ordinates, f_int (np array)
    '''
    import numpy as np
    # refinment factor for the interpolant. (If there are originally 10 points
    # but you want the interpolation to be defined on 50 points, then the 
    # refinement factor is 5.)
    refine_fac = 4
    # number of points passed in
    n = len(t)  
    # set t_int, if it hasn't been given
    if type(t_int) == bool:
        n_int = refine_fac*(n)
        t_int = np.linspace(t[0],t[-1],n_int)
    else:
        n_int = len(t_int)
    # implementation of the formula
    delta_t = t[1]
    f_int = []
    for t_i in t_int:
        f_int.append(sum([f_k*np.sinc((1/delta_t)*(t_i-t_k)) for t_k,f_k in zip(t,f)]))
    return (t_int, f_int)
#-----------------------------------------------------------------------------# 
def fourierInterp_given_freqs(x, y, omegas, x_int=False):
    '''
    This function interpolates a given set of ordinates and abscissas with a
    Fourier series that uses a specific set of frequencies. The interpolation 
    is constructed using coefficients found for the cosine and sine terms by
    solving a linear system built up from the given abscissas and ordinates.
    N.B. For this subroutine to work, the angular frequencies must be known 
        EXACTLY! Otherwise, the interpolant will only pass through the first
        N=2K+1 points (since the interpolant has only been found using those
        points). The subroutine isn't broken, you're just giving it bad 
        information.
    Input:
      - abscissas, x (as a list) (leave out last, duplicate point in period)
      - ordinates, y (as a list) (again, leave out last point, if periodic)
      - angular frequencies, omegas (as a list)
      - new abscissas, x_int (as a list) (optional! defaults to 10x refinement)
    Output:
      - new abscissas, x_int (as a list)
      - interpolated ordinates, y_int (as a list)
      - derivative of the interpolant, dydx_int (as a list)
    '''
    import math
    import numpy as np
    # refinment factor for the interpolant. (If there are originally 10 points
    # but you want the Fourier Series to be defined on 50 points, then the 
    # refinement factor is 5.)
    refine_fac = 10
    # number of points passed in
    n = len(x)                  
    # if zero has not been included as a frequency, add it
    if 0 not in omegas:
        omegas = [0.0] + omegas
    # total number of frequencies being considered, including the D.C. value
    K = len(omegas)-1
    # compute the coefficients by setting up and solving a linear system
    N = 2*K+1
    A = np.zeros((N,N))
    b = np.zeros((N,1))
    for i in range(N):
        b[i] = y[i]
        for j in range(N):
            if j == 0:
                A[i][j] = 1
            else:
                if j%2 == 1:
                    A[i][j] = math.cos(omegas[int((j+1)/2)]*x[i])
                else:
                    A[i][j] = math.sin(omegas[int(j/2)]*x[i])
    Fourier_coeffs = np.linalg.solve(A,b)
    # create separate lists for the cosine and sine coefficients
    a = []
    b = [0]
    for i in range(N):
        if i==0 or i%2==1:
            a.append(Fourier_coeffs[i][0])
        else:
            b.append(Fourier_coeffs[i][0])
    # set x_int, if it hasn't been given
    if type(x_int) == bool:
        n_int = refine_fac*(n)
        x_int = np.linspace(x[0],x[-1]+x[1],n_int)
    else:
        n_int = len(x_int)
    # find the actual interpolation
    y_int = [0.0]*n_int
    dydx_int = [0.0]*n_int
    for i in range(n_int):
        y_int[i] = a[0]        # the "DC" value
        dydx_int[i] = 0.0      # intialize the summation
        for j in range(K):
            y_int[i] += a[j+1]*math.cos(omegas[j+1]*x_int[i]) + \
                        b[j+1]*math.sin(omegas[j+1]*x_int[i])
            dydx_int[i] += omegas[j+1]* \
                           (b[j+1]*math.cos(omegas[j+1]*x_int[i]) - \
                            a[j+1]*math.sin(omegas[j+1]*x_int[i]))
    return (x_int, y_int, dydx_int)
#-----------------------------------------------------------------------------#
def my_dft(t, f, percent_energy_AC_peaks, shift_frequencies=False, 
           use_angular_frequencies=False, plot_spectrum=False, 
           plot_log_scale=False, refine_peaks=False, auto_open_plot=False,
           verbose=True):
    '''
    takes in a discrete signal and corresponding time points as numpy arrays 
    and returns the s and F, which are the discrete frequency samples indexed
    from -N/2 to N/2 and the corresponding values of the DFT, respectively.
    options:
    -percent_energy_AC_peaks:   this number will determine what counts as a 
                                peak and what doesn't. discrete frequency 
                                values in the first half of the spectrum (NOT 
                                including the D.C. value -- only A.C. content 
                                is considered) with the highest power are 
                                chosen, in descending order, to be peaks. peak 
                                selection ends once the selected peaks contain 
                                more than the specificed percentage of the 
                                total energy in the discrete signal, as given 
                                by Parseval's Theorem. note that since the 
                                power spectrum is mirrored, in practice, you're 
                                looking only at the positive discrete 
                                frequencies in the shifted DFT. the "total" 
                                energy should be computed only over these 
                                "postive frequencies by convention"
    -shift_frequencies:         if you want to generate values of the power   
                                spectrum spanning halfway into the negative 
                                frequencies
    -plot_angular_frequencies:  if, instead of using the discrete frequency 
                                samples, s, for plotting, use the corresponding
                                values of omega=2*pi*s
    -plot_spectrum:             if you want to create a plot of the power 
                                spectrum alongside the sampled points
    -plot_log_scale:            plot the spectrum on a logaritmic scale, such
                                that very low powers are magnified
    -refine_peaks:              take the peaks of the spectrum (defined as the 
                                highest-power A.C. components that capture the
                                desired signal energy) and, if any peaks are 
                                right next to each other, then view them as a 
                                single peak "cluster." define a new peak value 
                                that is the center of mass of the peak cluster. 
                                after that, create "bins" for each positive 
                                peak, starting at zero and going all the way to
                                the folding frequency. the bin boundaries are 
                                defined as halfway between peaks. these 
                                boundaries are then adjusted such that the 
                                boundary that is farther away from the 
                                clustered peak is brought closer to match the 
                                distance of the closer one. the refined peak 
                                values are then taken to be the centers of mass 
                                of each bin
    -auto_open_plot:            automatically open the saved figure of the 
                                signal and spectrum (w/peak refinement process)
    -verbose:                   print information about the spectrum and the 
                                refinement process to the screen
    Inputs:
        - t = time points of the discrete signal
        - f = signal values of the discrete signal
    Output:
        - the discrete frequencies on which the DFT is defined
        - the values of the DFT at those frequencies
        - the spectral peaks found (refined, if desired)
        - the adjusted boundaries of the peak clusters
    '''
    import numpy as np
    from matplotlib import pyplot as plt
    import webbrowser
    # the number of samples
    N = len(t)
    # figure out the entire signal duration (the samples are just shy of this)
    delta_t = t[1]-t[0]
    L = delta_t*N
    # compute the sampling rate being used (samples/seconds)
    sampling_rate_used = 1.0/delta_t
    # compute the DFT
    F = np.zeros((N,1), dtype='complex_')
    for m in range(N):
        F[m] = 0.0
        for n in range(N):
            F[m] += f[n]*np.exp(-2.0*np.pi*1j*m*n/N)
    # convert the DFT to powers
    powers = np.absolute(F)**2
    # compute 2B (the bandwidth)
    B = N/(2.0*L)
    # calculate the Nyquist frequency (a.k.a. folding frequency)
    if use_angular_frequencies:
        folding_freq = 2.0*np.pi*B      # [rad/s]
    else:
        folding_freq = B                # [Hz]
    # compute the discrete frequency values at which the DFT is being computed
    s = np.linspace(0.0,2.0*B,N+1)[:-1]
    
    # pick out the positive entries of the shifted spectrum (n.b. this does NOT
    # include the D.C. component)
    if N%2==0:
        s_half_positive = s[1:int(N/2)+1]
        powers_half_positive = powers[1:int(N/2)+1]
    else:
        s_half_positive = s[1:int((N+1)/2)]
        powers_half_positive = powers[1:int((N+1)/2)]
    # compute the total energy contained in the positive half of the spectrum
    N_half = len(powers_half_positive)
    E_half_positive = (1/N_half)*sum(powers_half_positive)
    # zip together the frequencies and powers, then sort them in descending 
    # order based on the power values (n.b. zipped object dies after sorting)
    points_half_positive = zip(s_half_positive, powers_half_positive)
    sorted_points = sorted(points_half_positive, 
                           key=lambda point: point[1], 
                           reverse=True)
    # starting from the end of the list, declare points to be peaks until the 
    # cumulative peak energy just exceeds the desired percentage
    positive_peaks = []
    cummulative_E = 0.0
    required_E = (percent_energy_AC_peaks/100.0)*E_half_positive
    last_point = False
    for point in sorted_points:
        # extract the two parts of the tuple
        point_s = point[0]
        point_power = point[1]
        # convert to rad/s, if desired
        if use_angular_frequencies:
            point_freq = 2.0*np.pi*point_s
        else:
            point_freq = point_s
        # add this power value to the cummulative sum (multiply by 1/N_half
        # here -- not 1/"# peaks recorded" -- because we're still in fact 
        # computing half the total energy of the signal, but now just 
        # pretending that the powers associated with all points yet to be 
        # selected are zero)
        cummulative_E += (1/N_half)*point_power
        # record this point as a peak
        positive_peaks.append((point_freq,point_power))
        # note if we're at the at end of the list (for setting "cutoff")
        if point == sorted_points[-1]:
            last_point = True
        # check to see if we've surpassed the required energy criterion
        if cummulative_E >= required_E:
            # compute the percent of the total energy captured by these peaks
            percent_E_captured = 100.0*(cummulative_E/E_half_positive)
            break
    # sort the newly found peaks into ascending order by frequency
    positive_peaks = sorted(positive_peaks, key=lambda point: point[0])
    # define a "cutoff" power (for the purposes of plotting) as the average of 
    # the power value of the last peak selected and the power value of the next 
    # point that was to have been evaluated. if there are no more points left
    # (every point available has been converted to a peak), then take an 
    # average of the last peak's power and zero. i.e. the cutoff is just half
    # the last peak's power
    if last_point:
        effective_power_cutoff = point_power/2.0
    else:
        # recover the index of the last recorded peak
        point_index = sorted_points.index(point)
        # extract the power of the next available point
        next_point_power = sorted_points[point_index+1][1]
        effective_power_cutoff = 0.5*(point_power + next_point_power)
    # if D.C. power exceeds this cutoff, then include it as a peak for printing
    DC_power = powers[0]
    DC_is_a_peak = False
    if DC_power >= effective_power_cutoff:
        DC_is_a_peak = True
    # pick out the positive frequency values (will be returned to calling code)
    positive_freqs = [peak[0] for peak in positive_peaks]    
    # if refining peaks and the spectrum isn't already being shifted, then
    # shift it so that you're only looking at the positive peaks
    if refine_peaks:
        shift_frequencies = True        
    # shift "negative frequencies by convention" onto real negative frequencies
    if shift_frequencies:
        if N%2==0:
            # even N
            s = np.hstack((np.fliplr([-s[1:int(N/2)]])[0], s[:int(N/2)+1]))
            F = np.vstack((F[int(N/2)+1:], F[:int(N/2)+1]))
            powers = np.vstack((powers[int(N/2)+1:], powers[:int(N/2)+1]))
        else:
            # odd N
            s = np.hstack((np.fliplr([-s[1:int((N+1)/2)]])[0], s[:int((N+1)/2)]))
            F = np.vstack((F[int((N+1)/2):], F[:int((N+1)/2)]))
            powers = np.vstack((powers[int((N+1)/2):], powers[:int((N+1)/2)]))
    # set the frequency plotting vector to the desired type
    if use_angular_frequencies:
        # compute the angular frequencies corresponding to s
        omega = 2.0*np.pi*s
        # set the frequency axis
        freqs = omega
        # compute the interval in the frequency domain
        delta_freq = 2.0*np.pi*(1.0/L)
        # write down the equation used to find delta_freq
        delta_freq_eq = '2*pi/L'
        # set the units label
        freq_label = 'radians/second'
        if refine_peaks:
            # find the angular frequencies corresponding to the first half of s 
            omega_half_positive = 2.0*np.pi*s_half_positive
            # set the postive frequencies, for peak refinement
            freqs_half_positive = omega_half_positive
    else:
        # set the frequency axis
        freqs = s
        # compute the interval in the frequency domain
        delta_freq = 1.0/L
        # write down the equation used to find delta_freq
        delta_freq_eq = '1/L'
        # set the units label
        freq_label = 'Hertz'
        if refine_peaks:
            # set the postive frequencies, for peak refinement
            freqs_half_positive = s_half_positive
    # set an internal value for being "close to zero"
    machine_zero = np.finfo(float).eps      # machine zero
    close_to_zero = machine_zero*1e3        # might as well be machine zero
    # check to see if any of the powers are zero (or close to machine zero)
    powers_close_to_zero = [power for power in powers if np.absolute(power) < close_to_zero]        
    # initialize the vector that will be returned, in case refinement is off
    peak_bounds_tuples = []    
    # if desired, begin refinement of the peaks
    if refine_peaks:
        if positive_peaks:
            # cluster the peaks that are only separated by delta_freq
            clusters = []
            current_cluster = []
            adjacent_peaks = False
            for i in range(len(positive_peaks)-1):
                current_freq = positive_peaks[i][0]
                next_freq = positive_peaks[i+1][0]
                distance_to_next_freq = next_freq-current_freq
                if abs(distance_to_next_freq-delta_freq) < close_to_zero:
                    # turn the flag on for having found adjacent peaks
                    adjacent_peaks = True                        
                    # append the first of the two peak tuples being compared
                    current_cluster.append(positive_peaks[i])
                else:
                    # turn the flag off for having found adjacent peaks
                    adjacent_peaks = False
                    # append the final peak tuple of the current cluster
                    current_cluster.append(positive_peaks[i])
                    # record this cluster into the list of clusters
                    clusters.append(current_cluster)
                    # clear the list holding the current cluster
                    current_cluster = []
            # deal with the case when the final peak is part of a cluster
            if adjacent_peaks:
                current_cluster.append(positive_peaks[-1])
                clusters.append(current_cluster)
            else:
                clusters.append([positive_peaks[-1]])
            # count up the number clusters found
            n_clusters = len(clusters)
            # compute each cluster's center of mass
            cluster_COMs_x = []             # x-coordinate of COM
            cluster_COMs_y = []             # y-coordinate of COM
            for cluster in clusters:
                # extract the power values in this cluster
                cluster_powers = [peak[1] for peak in cluster]
                # compute the total "mass"
                total_cluster_power = sum(cluster_powers)
                # compute the frequency coordinate of this cluster's COM
                weighted_sum_x = sum([peak[0]*peak[1] for peak in cluster])                
                cluster_COM_x = (1.0/total_cluster_power)*weighted_sum_x
                cluster_COMs_x.append(cluster_COM_x)
                # this part is only for plotting the y-coordinate of the COM
                weighted_sum_y = sum([peak[1]*peak[1] for peak in cluster])
                cluster_COM_y = (1.0/total_cluster_power)*weighted_sum_y
                cluster_COMs_y.append(cluster_COM_y)
            # set the boundaries, the first bin starts at 0.0
            boundaries = [0.0]
            # if there's more than one clustered peak frequency... 
            if n_clusters > 1:
                # bound a given peak's bin halfway to the next peak
                for i in range(n_clusters-1):
                    boundaries.append(0.5*(cluster_COMs_x[i]+cluster_COMs_x[i+1]))
            # the last bin ends at the folding frequency
            boundaries.append(folding_freq)
            # put into "peak bins" the frequencies within its boundaries
            refined_positive_freqs = []             # x-coordinate of COM
            refined_COMs_y = []                     # y-coordinate of COM
            clustered_freq_boundaries = []
            peak_bounds_tuples = []                 # list will be returned
            for i in range(n_clusters):
                # extract the current clustered peak frequency
                clustered_freq = cluster_COMs_x[i]
                # when the number of sample points used is even, the folding 
                # frequency always coincides with the (N/2+1)-th discrete 
                # frequency in the spectrum. if it so happens that the folding 
                # frequency is also a lone peak (i.e. a cluster of one), then 
                # the refinement procedure will not work, since the shortest 
                # distance to a bin boundary is zero. (recall that the folding 
                # frequency is always the upper boundary of the last bin.) so,
                # the "adjusted" lower and upper boundaries are just the 
                # folding frequency itself and the width of the bin is zero. in
                # this case, there are no points over which the peak can be 
                # refined (no points over which we can compute a center of 
                # mass). so, the refined peak is just the peak itself, i.e. the
                # folding frequency
                if clustered_freq == folding_freq:
                    refined_positive_freqs.append(folding_freq)
                    folding_power = powers_half_positive[-1]
                    refined_COMs_y.append(folding_power)
                    # the tuple of (lower_boundary, upper_boundary) for this
                    # case will just be (folding_freq, folding_freq). N.B. if
                    # this case occurs, it will always be the last one in the
                    # list, since the clustered peaks are already sorted in 
                    # ascending order
                    peak_bounds_tuples.append((folding_freq,folding_freq))
                # if not dealing with special case, then start refinement
                else:
                    # extract the upper and lower boundary of this peak's bin
                    lower_boundary_bin = boundaries[i]
                    upper_boundary_bin = boundaries[i+1]
                    # adjust the boundaries so that each clustered peak 
                    # frequency is equidistant from its boundaries. set this 
                    # distance equal to the smaller of the clustered peak's 
                    # distances from its current boundaries
                    dist_to_lower = clustered_freq - lower_boundary_bin
                    dist_to_upper = upper_boundary_bin - clustered_freq
                    smaller_dist = min(dist_to_lower, dist_to_upper)
                    lower_boundary = clustered_freq - smaller_dist
                    upper_boundary = clustered_freq + smaller_dist
                    # convert the values to floats from 1-element numpy arrays
                    lower_boundary = lower_boundary[0]
                    upper_boundary = upper_boundary[0]
                    # record these adjusted boundaries (for plotting)
                    clustered_freq_boundaries.append(lower_boundary)
                    clustered_freq_boundaries.append(upper_boundary)
                    # append these boundies to the list that will be returned
                    peak_bounds_tuples.append((lower_boundary,upper_boundary))
                    # determine which discrete frequencies fall within this bin
                    included_freqs = []
                    included_powers = []
                    for j in range(len(freqs_half_positive)):
                        current_freq = freqs_half_positive[j]
                        current_power = powers_half_positive[j]
                        if current_freq > lower_boundary and current_freq <= upper_boundary:
                            included_freqs.append(current_freq)
                            included_powers.append(current_power)
                    # for nomralization of the weights (powers)
                    total_included_power = sum(included_powers)
                    # compute the weighted sum of the included powers
                    weighted_sum = 0.0
                    for j in range(len(included_freqs)):
                        freq = included_freqs[j]
                        power = included_powers[j]
                        weighted_sum += power*freq
                    # compute center of mass (weighted average) (refined peak)
                    refined_freq = (1.0/total_included_power)*weighted_sum
                    # record the center of mass (which is the refined peak)
                    refined_positive_freqs.append(refined_freq[0])
                    # compute the y-coordinate of the center of mass (plotting)
                    weighted_sum = 0.0
                    for j in range(len(included_freqs)):
                        power = included_powers[j]
                        weighted_sum += power*power
                    # find the y-coordinate of this center of mass
                    refined_COM_y = (1.0/total_included_power)*weighted_sum
                    # record the coordinate of center of mass (for plotting)
                    refined_COMs_y.append(refined_COM_y)
        else:
            print('\n\tERROR:  No peaks found...something is wrong. \n\t\t' + \
                  'Increase percentage of total energy desired in peaks. ' + \
                  '\n\t\tCurrent value: '+str(percent_energy_AC_peaks)+'%.\n')
            return()
    # if not refining peaks, set that variable equal to the positive peaks
    if not refine_peaks:
        refined_positive_freqs = positive_freqs
    # print the salient quantities to the console, if desired
    if verbose:
        print('\n'+'-'*75)
        print('\n\tfundamental relations:')
        print('\n\t\tN = 2*B*L')
        print('\t\tL = N*delta_t')
        print('\n\t\tN =', round(N,2), 'samples \t\t (no. of time samples given)')
        print('\t\tL =', round(L,2), 'seconds \t (one interval beyond last point)')
        print('\t\tB =', round(B,2), 'Hertz \t\t (1/2 the full bandwidth captured)')
        print('\n\t\tdelta_t = 1/(2*B): \t', round(1.0/(2.0*B),3), 'seconds')
        print('\t\tdelta_freq =', delta_freq_eq+':\t', round(delta_freq,3), freq_label)
        print('\n\n\tsampling rate used: \t\t ', round(sampling_rate_used,2), 'samples/second')
        print('\t\t\t\t\t  ('+str(round(2.0*np.pi*sampling_rate_used,2))+' radians/second)')
        print('\n\tNyquist ("folding") frequency:\t ', round(folding_freq,2), freq_label)
        print('\n\t'+str(len(positive_peaks)) + ' spectral peaks')
        print('\t(capturing '+str(round(percent_E_captured[0],2))+'% of A.C. energy):')
        for peak in positive_peaks:
            print('\t\t\t\t\t ', round(peak[0],2), freq_label)
        if refine_peaks:
            print('\t'+str(len(refined_positive_freqs)) + ' refined spectral peaks:')
            for refined_freq in refined_positive_freqs:
                print('\t\t\t\t\t ', round(refined_freq,2), freq_label)
        print('\n'+'-'*75)
    # plot the spectrum    
    if plot_spectrum:
        # plotting preliminaries
        plot_name = 'the_DFT'
        auto_open = auto_open_plot
        plt.figure(plot_name,figsize=(12,5))
        # plot the signal
        plt.subplot(1,2,1)
        plt.plot(t,f,'ko-',label='$f(t)$')
        plt.xlabel('$t, [s]$', fontsize=16)
        plt.ylabel('$f(t)$', fontsize=16)
        plt.legend(loc='best',fontsize=12)
        # plot the DFT
        plt.subplot(1,2,2)
        # set the y position from which the vertical lines start
        if powers_close_to_zero:
            vlines_bottom = 0.0
            plot_log_scale=False
            # print a message to the screen if can't plot on log scale
            if len(powers_close_to_zero) == 1:
                value_string = 'VALUE EXISTS'
            else:
                value_string = 'VALUES EXIST'
            print('\n\tN.B. CAN\'T PLOT ON LOG SCALE BECAUSE '+str(len(powers_close_to_zero))+' \n\t' + \
                  '     ZERO '+value_string+' IN THE POWER SPECTRUM!\n\t' + \
                  '     PLOTTING SPECTURM ON LINEAR AXIS INSTEAD...')
        else:
            vlines_bottom = 10**np.floor(np.log10(min(powers)))
        # plot the powers on the desired y-axis
        if plot_log_scale:
            plt.semilogy(freqs,powers,'ko',label='$power \, spectrum$')
        else:
            plt.plot(freqs,powers,'ko',label='$power \, spectrum$')
        # plot the vertical lines
        plt.vlines(freqs,[vlines_bottom]*N,powers,'k')
        # plot the power cutoff
        plt.plot([0.0,folding_freq],[effective_power_cutoff]*2,'y--', label='$peak \,\, cutoff$')
        # plot the Nyquist frequency and add the appropriate axis labels
        if use_angular_frequencies:
            folding_label = '$\omega_{nyquist}$'
            plt.xlabel('$\omega, [\\frac{rad}{s}]$', fontsize=16)
            plt.ylabel('$|\mathcal{F}f(\omega)|^2$', fontsize=16)
        else:
            folding_label = '$s_{nyquist}$'
            plt.xlabel('$s, [Hz]$', fontsize=16)
            plt.ylabel('$|\mathcal{F}f(s)|^2$', fontsize=16)
        # if refining, plot the relevant quantities 
        if refine_peaks:
            # plot the bin boundaries
            bin_boundary_color = 'g'
            for boundary in boundaries:
                if boundary==boundaries[-1]:
                    # plot a solid yellow line at the folding frequency
                    plt.plot([boundary]*2,[vlines_bottom,max(powers)], bin_boundary_color)
                elif boundary in clustered_freq_boundaries:
                    # at the points where clustered-peak boundaries overlap the 
                    # orginal bin boundaries, plot a solid vertical line
                    plt.plot([boundary]*2,[vlines_bottom,max(powers)], bin_boundary_color)
                else:
                    if boundary==boundaries[0]:
                        # plot the first boundary and add to the legend
                        plt.plot([boundary]*2,[vlines_bottom,max(powers)], bin_boundary_color+'--',label='$bin \,\, boundaries$')
                    else:
                        # plot the remaining boundaries
                        plt.plot([boundary]*2,[vlines_bottom,max(powers)], bin_boundary_color+'--')
            # plot the adjusted, clustered peak boundaries
            for boundary in clustered_freq_boundaries:
                if boundary==clustered_freq_boundaries[0]:
                    # plot the first peak boundary and add to the legend
                    plt.plot([boundary]*2,[vlines_bottom,max(powers)],'m--',label='$cluster \,\, boundaries$')
                else:
                    # plot the remaining boundaries
                    plt.plot([boundary]*2,[vlines_bottom,max(powers)],'m--')
            # plot the centers of mass corresponding to each peak cluster
            plt.plot(cluster_COMs_x, cluster_COMs_y,'b*')
            plt.plot(refined_positive_freqs, refined_COMs_y,'m*')
            # draw ellipses around clusters
            one_cluster_plotted = False
            cluster_counter = 0
            for cluster in clusters:
                n_included_peaks = len(cluster)
                # exclude clusters that are just lonesome peaks
                if n_included_peaks > 1:
                    # extract the included frequencies and powers
                    included_freqs = [peak[0] for peak in cluster]
                    included_powers = [peak[1] for peak in cluster]
                    # set the semi-minor axis in the x direction
                    semi_minor_axis = 1.5*(included_freqs[-1]-included_freqs[0])
                    # set the semi-major axis in the y direction
                    semi_major_axis = max(included_powers)-min(included_powers)
                    # set the center of the ellipse
                    x_center = sum(included_freqs)/n_included_peaks
                    y_center = sum(included_powers)/n_included_peaks
                    # draw out the ellipse using polar coordinates
                    thetas = np.linspace(0.0, 2.0*np.pi, 3*36)
                    x_ellipse = []
                    y_ellipse = []
                    for theta in thetas:
                        # compute points and convert to cartesian coordinates
                        x_point = x_center + semi_minor_axis*np.cos(theta)
                        y_point = y_center + semi_major_axis*np.sin(theta)
                        # if the x-point falls outside the adjusted cluster
                        # boundaries, then reassign that point to the value of
                        # this nearest boundary
                        lower_boundary = peak_bounds_tuples[cluster_counter][0]
                        upper_boundary = peak_bounds_tuples[cluster_counter][1]
                        if x_point < lower_boundary:
                            x_ellipse.append(lower_boundary)
                        elif x_point > upper_boundary:
                            x_ellipse.append(upper_boundary)
                        else:
                            x_ellipse.append(x_point)
                        # if the y-point is greater that where the vertical 
                        # lines start, then record it, otherwise, ignore
                        if y_point >= effective_power_cutoff:
                            y_ellipse.append(y_point)
                        else:
                            y_ellipse.append(effective_power_cutoff)
                    # plot the ellipse
                    if one_cluster_plotted:
                        plt.plot(x_ellipse, y_ellipse, 'b-')
                    else:
                        plt.plot(x_ellipse, y_ellipse, 'b-', label='$peak \,\, clusters$')
                        one_cluster_plotted = True
                # increment counter
                cluster_counter += 1
        # plot the Nyquist frequecy (a.k.a folding frequency)
        plt.plot([folding_freq]*2,[vlines_bottom,max(powers)],'r--',label=folding_label)
        # write the plot title with the peaks    
        title = '$peaks: \quad'
        if DC_is_a_peak:
            title += str('0.00, \,')
        for freq in refined_positive_freqs:
            title += str(np.round(freq,2)) 
            if freq != refined_positive_freqs[-1]:
                title += ', \,'
        title += '$'
        # plot the title and the legend
        plt.title(title)
        plt.legend(loc='best',fontsize=12)
        # use "tight layout"
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
    # return the peaks
    return freqs, powers, refined_positive_freqs, peak_bounds_tuples
#-----------------------------------------------------------------------------#    
def main():
    '''
    main function, executed when run as a standalone file
    '''

    # test the DFT
    omegas_actual = [5.2, 7.93]
    
    # find the sampling rate needed (Nyquist rate) to capture all frequencies used
    max_omega = max(omegas_actual)
    max_s = max_omega/(2.0*np.pi)
    nyquist_rate = 2.0*max_s
    nyquist_rate_rad = 2.0*np.pi*nyquist_rate
    delta_t_needed = 1.0/nyquist_rate
    
    # generate a discrete signal over a small set of points
    t_start = 0.0
    t_end = 10.0
    n_points = 26
    t = np.linspace(t_start, t_end, n_points)
    f = sum([np.sin(omega*t) + 2.0*np.cos(omega*t) for omega in omegas_actual]) + 6.0
    # define the same signal on a very fine grid (to get the "exact" function)
    t_fine = np.linspace(t_start, t_end, 100*n_points)
    f_fine = sum([np.sin(omega*t_fine) + 2.0*np.cos(omega*t_fine) for omega in omegas_actual]) + 6.0
    
    # find the sinc interpolation of the time signal
    t_fine, f_int = whittaker_shannon_interp(t,f,t_fine)
    
    # take the DFT to find the peaks as best you can
    s, F, peaks_found, peak_boundaries = my_dft(t, f, 
                                                 percent_energy_AC_peaks=97,
                                                 shift_frequencies=True,
                                                 use_angular_frequencies=True,
                                                 plot_spectrum=True, 
                                                 plot_log_scale=True,
                                                 refine_peaks=True,
                                                 auto_open_plot=True,
                                                 verbose=True)
    
    # interpolate the points using the tonal peaks found by taking the DFT
    t_fine, f_int_tones, dummy = fourierInterp_given_freqs(t, f, peaks_found, t_fine)
    
    # interpolate the points using the correct omegas
    t_fine, f_int_omegas, dummy = fourierInterp_given_freqs(t, f, omegas_actual, t_fine)
    
    # print important quantities
    print('\n\tactual frequencies:')
    for omega in omegas_actual:
        print('\t\t\t\t\t ', round(omega,2), 'radians/second')
    print('\n\tnyquist rate \n\t(i.e. sampling rate needed):     ', \
           round(nyquist_rate,2), 'samples/second')
    print('\t\t\t\t\t  ('+str(round(nyquist_rate_rad,2))+' radians/second)')
    print('\n\tdelta_t needed: \t\t ', round(delta_t_needed,2), 'seconds')
    
    # plot signals and interpolations
    linewidth = 2.0
    plt.close('all')
    plt.figure('interpolation', dpi=110)
    plt.plot(t_fine,f_fine,'k-',linewidth=linewidth)
    plt.plot(t,f,'ko',label='$f \,\, samples$',markersize=8)
    plt.plot(t_fine,f_int,'b--',linewidth=linewidth,label='$sinc \,\, interp.$')
    plt.plot(t_fine,f_int_tones,'r--',linewidth=linewidth,label='$trig. \,\, interp \,\, w/\mathbf{\omega}_{found}$')     
    plt.plot(t_fine,f_int_omegas,'y--',linewidth=linewidth,label='$trig. \,\, interp \,\, w/\mathbf{\omega}_{actual}$')     
    plt.xlabel('$t, [s]$', fontsize=16)
    plt.ylabel('$f(t)$', fontsize=16)
    plt.legend(loc='best',fontsize=16)
#-----------------------------------------------------------------------------#

# standard boilerplate to call the main() function
if __name__ == '__main__':
    main()