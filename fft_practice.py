# -*- coding: utf-8 -*-
"""
Created on Mon May 16 14:44:22 2016

@author: Kedar R. Naik
"""
import csv
import numpy as np
from matplotlib import pyplot as plt
plt.close('all')

# if you don't have a CSV file full of points, define your own signal
points = 500
t = np.linspace(0,2*np.pi,points)
f = np.cos(2*t) + 1.0*np.sin(50*t) + 1.0*np.sin(70*t)

# otherwise, provide some details about the CSV file and set flag to True
signal_from_CSV = False
csv_filename = 'f'
analyze_column = 'Close'

# in either case, set some constraints on the reconstruction
energy_cutoff = 1e2
use_K_frequencies = 20       # not including the DC component


if signal_from_CSV:
    # read in the results file
    history = []
    with open(csv_filename+'.csv','r') as history_file:
        history_reader = csv.reader(history_file)
        for row in history_reader:
            history.append(row)
    # remove the header row
    header = history.pop(0)
    print('\n\tCSV file read in')
    
    # figure out which columnn contains the desired data
    column_counter = 0
    for entry in header:
        if entry == 'Date':
            date_index = column_counter
        if entry == analyze_column:
            index = column_counter
            break
        column_counter += 1
    
    # extract the date and column to be analyzed from the data
    dates = np.array([row[date_index] for row in history])
    data = np.flipud(np.array([row[index] for row in history]))
    print('\n\tdata extracted')
    
    # define the signal in the time domain
    t = dates
    f =  data
    t = np.linspace(0,len(f),len(f))
print('\n\tsignal defined')

# take the Fourier transform and compute the energy spectrum
Ff = np.fft.fft(f)
energies = np.absolute(Ff)**2
print('\n\tDFT and energy spectrum computed')

# filter the spectrum by zeroing out everthing below the cutoff
Ff_filtered = [Ff[0]] # always include the DC component
retained_entry_counter = 1
for i in range(1,len(energies)): 
    if energies[i] > energy_cutoff:
        Ff_filtered.append(Ff[i])
        retained_entry_counter += 1
    else:
        Ff_filtered.append(0.0)
Ff_filtered = np.array(Ff_filtered)
energies_filtered = np.absolute(Ff_filtered)**2
cutoff_active = True

# if the number of active frequencies (aside from the DC component) larger than 
# the maximum frequecies desired to be retained. zero out all but the 
# 2*use_K_frequencies spectrum points with the highest energy
if 2*use_K_frequencies < retained_entry_counter-1:
    cutoff_active = False
    sorted_filtered_energies = np.sort(energies_filtered[1:])
    energies_to_retain = sorted_filtered_energies[-(2*use_K_frequencies):]
    min_retained_energy = energies_to_retain[0]
    for i in range(1,len(energies)):
        if energies[i] < min_retained_energy:
            Ff_filtered[i] = 0.0
energies_filtered = np.absolute(Ff_filtered)**2

# take the inverse Fourier transform of the filtered Fourier transform
f_recon = np.fft.ifft(Ff_filtered)

# the reconstructed signal is real, but python returns some points with very 
# small complex components. cast those values as real if they're within three 
# orders of magnitude away from machine zero
f_recon = np.real_if_close(f_recon, tol=1000)

# print some information about the reconstruction to the screen
if cutoff_active:
    print('\n\treconstruction limited by min. energy cutoff:',energy_cutoff)
else:
    print('\n\treconstruction limited by number of retained frequencies:', use_K_frequencies)

# plot the signal
plt.figure()
plt.subplot(1,2,1)
plt.plot(t,f,'b-')
plt.xlabel('$t$', fontsize=16)
plt.ylabel('$f(t)$', fontsize=16)

# plot the energy spectrum
plt.subplot(1,2,2)
freqs = np.fft.fftfreq(len(f),1/points)
plt.semilogy(np.fft.fftshift(freqs),np.fft.fftshift(energies),'bo')
plt.vlines(np.fft.fftshift(freqs),[10**np.floor(np.log10(min(energies)))]*len(freqs),np.fft.fftshift(energies),'b')
plt.xlabel('$\omega$', fontsize=16)
plt.ylabel('$|Ff(\omega)|^2$', fontsize=16)

# plot the energy spectrum of the filtered Fourier transform
plt.subplot(1,2,2)
freqs = np.fft.fftfreq(len(f),1/points)
plt.semilogy(np.fft.fftshift(freqs),np.fft.fftshift(energies_filtered),'ro')
plt.vlines(np.fft.fftshift(freqs),[10**np.floor(np.log10(min(energies)))]*len(freqs),np.fft.fftshift(energies_filtered),'r')
if cutoff_active:
    plt.semilogy(np.fft.fftshift(freqs),[energy_cutoff]*len(energies),'r--')
plt.xlabel('$\omega$', fontsize=16)
plt.ylabel('$|Ff(\omega)|^2$', fontsize=16)

# plot the reconstructed signal
plt.subplot(1,2,1)
plt.plot(t,f_recon,'r-')
plt.xlabel('$t$', fontsize=16)
plt.ylabel('$f(t)$', fontsize=16)
plt.tight_layout()

