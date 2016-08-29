# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 19:24:44 2016

@author: Kedar
"""
import numpy as np

#-----------------------------------------------------------------------------# 
def my_dft(t, f, shift_frequencies=False, use_angular_frequencies=False, 
           plot_spectrum=False, plot_log_scale=False, refine_peaks=False,
           auto_open_plot=False):
    '''
    takes in a discrete signal and corresponding time points as numpy arrays 
    and returns the s and F, which are the discrete frequency samples indexed
    from -N/2 to N/2 and the corresponding values of the DFT, respectively.
    options:
    -shift_frequencies: if you want to generate values of the energy spectrum  
                        spanning halfway into the negative frequencies
    -plot_spectrum: if you want to create a plot of the power spectrum
    -plot_angular_frequencies: if, instead of using the discrete frequency 
                        samples, s, for plotting, use the corresponding values 
                        of omega=2*pi*s 
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
    # convert the DFT to energies
    energies = np.absolute(F)**2
    # compute 2B (the bandwidth)
    B = N/(2.0*L)
    # calculate the Nyquist frequency (a.k.a. folding frequency)
    if use_angular_frequencies:
        folding_freq = 2.0*np.pi*B      # [rad/s]
    else:
        folding_freq = B                # [Hz]
    # compute the discrete frequency values at which the DFT is being computed
    s = np.linspace(0.0,2.0*B,N+1)[:-1]
    # split the discrete frequencies and pick out the positive entries
    if refine_peaks:
        if N%2==0:
            s_half_positive = s[1:int(N/2)+1]
            energies_half_positive = energies[1:int(N/2)+1]
        else:
            s_half_positive = s[1:int((N+1)/2)]
            energies_half_positive = energies[1:int((N+1)/2)]
        # if refining peaks and the spectrum isn't already being shifted, then
        # shift it so that you're only looking at the positive peaks
        shift_frequencies = True        
    # shift "negative frequencies by convention" onto real negative frequencies
    if shift_frequencies:
        if N%2==0:
            # even N
            s = np.hstack((np.fliplr([-s[1:int(N/2)]])[0], s[:int(N/2)+1]))
            F = np.vstack((F[int(N/2)+1:], F[:int(N/2)+1]))
            energies = np.vstack((energies[int(N/2)+1:], energies[:int(N/2)+1]))
        else:
            # odd N
            s = np.hstack((np.fliplr([-s[1:int((N+1)/2)]])[0], s[:int((N+1)/2)]))
            F = np.vstack((F[int((N+1)/2):], F[:int((N+1)/2)]))
            energies = np.vstack((energies[int((N+1)/2):], energies[:int((N+1)/2)]))
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
            # compute the angular frequencies corresponding to the first half of s 
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
    # check to see if any of the energies are zero (or close to machine zero)
    energies_close_to_zero = [energy for energy in energies if np.absolute(energy) < close_to_zero]   
    # set an energy cutoff that sets the bar for what counts as a peak
    energies_without_DC = [energies[i] for i in range(N) if freqs[i] != 0.0]
    energy_cutoff = np.average(energies_without_DC) + 0.0*np.std(energies_without_DC)
    # pick out the peaks (freq,energy) in the energy spectrum (list of tuples)
    peaks = [(freqs[i],energies[i][0]) for i in range(N) if energies[i] > energy_cutoff]
    # pick out the nonnegative peaks
    nonnegative_peaks = [peak for peak in peaks if peak[0] >= 0.0]
    # if desired, begin refinement of the peaks
    if refine_peaks:
        # find the boundaries of the "bins" corresponding to each peak
        if len(peaks) != 0:
            # count up the number of peaks found
            n_peaks = len(peaks)
            # if the DC value is a peak, the count will be odd
            if n_peaks%2 == 1:
                # remove the DC value from the count so that n_peaks is even
                n_peaks -= 1
            # if there are still peaks left, cluster them, bound them, bin them, refine them
            if n_peaks >= 2:
                # pull out the positive peaks
                positive_peaks = peaks[-int(n_peaks/2):]
                # cluster the peaks that are only separated by delta_freq
                clusters = []
                current_cluster = []
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
                        # turn the flag on for having found adjacent peaks
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
                # compute frequencies that are each cluster's center of mass
                clustered_freqs = []
                for cluster in clusters:
                    cluster_energies = [peak[1] for peak in cluster]
                    total_cluster_energy = sum(cluster_energies)
                    weighted_sum = sum([peak[0]*peak[1] for peak in cluster])
                    cluster_center_of_mass = (1.0/total_cluster_energy)*weighted_sum
                    clustered_freqs.append(cluster_center_of_mass)
                # set the boundaries, the first bin starts at 0.0
                boundaries = [0.0]
                # if there's more than one clustered peak frequency... 
                if len(clustered_freqs) > 1:
                    # bound a given peak's bin halfway to the next peak
                    for i in range(len(clustered_freqs)-1):
                        boundaries.append(0.5*(clustered_freqs[i]+clustered_freqs[i+1]))
                # the last bin ends at the folding frequency
                boundaries.append(folding_freq)
                # put into "peak bins" the frequencies within its boundaries
                refined_positive_freqs = []
                for i in range(len(clustered_freqs)):
                    lower_boundary = boundaries[i]
                    upper_boundary = boundaries[i+1]
                    included_freqs = []
                    included_energies = []
                    for j in range(len(freqs_half_positive)):
                        current_freq = freqs_half_positive[j]
                        current_energy = energies_half_positive[j]
                        if current_freq > lower_boundary and current_freq <= upper_boundary:
                            included_freqs.append(current_freq)
                            included_energies.append(current_energy)
                    # for nomralization of the weights (energies)
                    total_included_energy = sum(included_energies)
                    # compute the center of mass of the included energies
                    weighted_sum = 0.0
                    for j in range(len(included_freqs)):
                        freq = included_freqs[j]
                        energy = included_energies[j]
                        weighted_sum += energy*freq
                    # compute weighted average (a.k.a. the refined peak)
                    refined_freq = (1.0/total_included_energy)*weighted_sum
                    # record the center of mass (which is the refined peak)
                    refined_positive_freqs.append(refined_freq[0])
    # print the salient quantities to the console
    print('\n'+'-'*75)
    print('\n\tfundamental relations:')
    print('\n\t\tN = 2*B*L')
    print('\t\tL = N*delta_t')
    print('\n\t\tN =', round(N,2), 'samples \t\t (no. of time samples given)')
    print('\t\tL =', round(L,2), 'seconds \t (one interval beyond last point)')
    print('\t\tB =', round(B,2), 'Hertz \t\t (1/2 the full bandwidth captured)')
    print('\n\t\tdelta_t = 1/(2*B): \t', round(1.0/(2.0*B),3), 'seconds')
    print('\t\tdelta_freq =', delta_freq_eq+':\t', round(delta_freq,3), freq_label)
    print('\n\n\tsampling rate used: \t\t', round(sampling_rate_used,2), 'samples/second')
    print('\tNyquist ("folding") frequency:\t', round(folding_freq,2), freq_label)
    if refine_peaks:
        print('\n\t'+str(len(positive_peaks)) + ' spectral peaks:')
        for peak in positive_peaks:
            print('\t\t\t\t\t ', round(peak[0],2), freq_label)
        print('\t'+str(len(refined_positive_freqs)) + ' refined spectral peaks:')
        for refined_freq in refined_positive_freqs:
            print('\t\t\t\t\t ', round(refined_freq,2), freq_label)
    else:
        print('\n\t'+str(len(nonnegative_peaks)) + ' spectral peaks:')
        for peak in nonnegative_peaks:
            print('\t\t\t\t\t ', round(peak[0],2), freq_label)
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
        if energies_close_to_zero:
            vlines_bottom = 0.0
            plot_log_scale=False
            # print a message to the screen if can't plot on log scale
            if len(energies_close_to_zero) == 1:
                value_string = 'VALUE EXISTS'
            else:
                value_string = 'VALUES EXIST'
            print('\n\tN.B. CAN\'T PLOT ON LOG SCALE BECAUSE '+str(len(energies_close_to_zero))+' \n\t' + \
                  '     ZERO '+value_string+' IN THE POWER SPECTRUM!\n\t' + \
                  '     PLOTTING SPECTURM ON LINEAR AXIS INSTEAD...')
        else:
            vlines_bottom = 10**np.floor(np.log10(min(energies)))
        # plot the energies on the desired y-axis
        if plot_log_scale:
            plt.semilogy(freqs,energies,'ko',label='$power \, spectrum$')
        else:
            plt.plot(freqs,energies,'ko',label='$power \, spectrum$')
        # plot the vertical lines
        plt.vlines(freqs,[vlines_bottom]*n_points,energies,'k')
        # plot the energy cutoff
        plt.plot([0.0,folding_freq],[energy_cutoff]*2,'y--', label='$peak \,\, cutoff$')
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
            for boundary in boundaries:
                if boundary==boundaries[-1]:
                    # plot a solid yellow line at the folding frequency
                    plt.plot([boundary]*2,[vlines_bottom,max(energies)],'c-')
                else:
                    if boundary==boundaries[0]:
                        # plot the first boundary and add to the legend
                        plt.plot([boundary]*2,[vlines_bottom,max(energies)],'c--',label='$bin \,\, boundaries$')
                    else:
                        # plot the remaining boundaries
                        plt.plot([boundary]*2,[vlines_bottom,max(energies)],'c--')
            # draw ellipses around clusters
            one_cluster_plotted = False
            for cluster in clusters:
                n_included_peaks = len(cluster)
                # exclude clusters that are just lonesome peaks
                if n_included_peaks > 1:
                    # extract the included frequencies and energies
                    included_freqs = [peak[0] for peak in cluster]
                    included_energies = [peak[1] for peak in cluster]
                    # set the semi-minor axis in the x direction
                    semi_minor_axis = 2.0*(included_freqs[-1]-included_freqs[0])
                    # set the semi-major axis in the y direction
                    semi_major_axis = max(included_energies)-min(included_energies)
                    # set the center of the ellipse
                    x_center = sum(included_freqs)/n_included_peaks
                    y_center = sum(included_energies)/n_included_peaks
                    # draw out the ellipse using polar coordinates
                    thetas = np.linspace(0.0, 2.0*np.pi, 36)
                    x_ellipse = []
                    y_ellipse = []
                    for theta in thetas:
                        # convert to cartesian coordinates
                        x_ellipse.append(x_center + semi_minor_axis*np.cos(theta))
                        y_ellipse.append(y_center + semi_major_axis*np.sin(theta))
                    # plot the ellipse
                    if one_cluster_plotted:
                        plt.plot(x_ellipse, y_ellipse, 'b-')
                    else:
                        plt.plot(x_ellipse, y_ellipse, 'b-', label='$peak \,\, clusters$')
                        one_cluster_plotted = True
        # plot the Nyquist frequecy (a.k.a folding frequency)
        plt.plot([folding_freq]*2,[vlines_bottom,max(energies)],'r--',label=folding_label)
        # write the plot title with the peaks    
        title = '$peaks: \quad'
        for peak in nonnegative_peaks:
            title += str(np.round(peak[0],2)) 
            if peak[0] != peaks[-1][0]:
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
    # return the relevant quantities
    return freqs, energies, sampling_rate_used
#-----------------------------------------------------------------------------#    

# test the DFT
omegas_actual = [5.0, 8.43]
t_start = 0.0
t_end = 10.0
n_points = 31
t = np.linspace(t_start, t_end, n_points)
f = sum([np.sin(omega*t) for omega in omegas_actual]) + 6.0
 
freqs, energies, sampling_rate_used = my_dft(t, f, 
                                         shift_frequencies=True,
                                         use_angular_frequencies=True,
                                         plot_spectrum=True, 
                                         plot_log_scale=True,
                                         refine_peaks=True,
                                         auto_open_plot=True)
