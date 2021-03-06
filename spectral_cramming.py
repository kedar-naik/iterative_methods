# -*- coding: utf-8 -*-
"""
This script examines what happens when you try to "cram" a broadband spectrum
into a few discrete tonal frequencies.


Created on Mon Oct  5 21:32:10 2015

@author: Kedar Naik
"""
import platform
import numpy as np
from matplotlib import pyplot as plt

#plt.close('all')

# let the user know that you're using Python 3
print('Using Python version ', platform.python_version())


###############################################################################
def my_dft(f, freqs=[-1.5]):
    '''
    This function returns the discrete Fourier transform of a given signal
    '''
    N = len(f)
    if freqs[0] == -1.5:
        freqs = list(range(N))
        F = np.zeros(N, dtype=complex)
    else:
        F = np.zeros(len(freqs), dtype=complex)
    index = 0
    for n in freqs:
        for k in list(range(N)):
            F[index] += f[k]*np.exp(-2.0*np.pi*1j*n*k/N)
        index += 1
    return F
    
###############################################################################


# zoom percentage for the frequency domain plots
zoom_percent = 10

# x-axis of frequency plot
freq_axis = 's'    # ('omega' OR 's') 

# number of sample points
N = pow(2,8)

# the signal
T = 2.0
t = np.linspace(0,T,N)
f = np.sin(2.0*np.pi*2.0*t/T) \
  + np.cos(2.0*np.pi*5.0*t/T) #  + np.sin(2.0*np.pi*80.0*t/T)

# the angular frequency
angular_frequency = 2.0*np.pi/T

# specific frequencies
specific_frequencies = [2.1, 5.1]
#specific_frequencies = [angular_frequency, 2*angular_frequency]

# the DFT
#F_f = np.fft.fft(f)
F_f = my_dft(f)

# plot the DFT
freq_max = int((zoom_percent/100.0)*N)
freq = list(range(0,freq_max))
if freq_axis == 's':
    freq_label = '$s$'
    F_label = '$\|Ff(s)\|^2$'
if freq_axis == 'omega':
    freq = [entry/angular_frequency for entry in freq]
    freq_label = '$\omega$'
    F_label = '$\|Ff(\omega)\|^2$'
    
# compute the inverse DFT
F_inv_F = np.fft.ifft(F_f)

# plot the signal in the time domain
plt.figure(1)
plt.clf()
plt.plot(t, f, label='$f$')
plt.xlabel('$t$')
plt.ylabel('$f(t)$')

# plot the frequency domain
plt.figure(2)
plt.clf()
plt.plot(freq,np.power(np.absolute(F_f[0:freq_max]),2),'ko')
plt.vlines(freq,np.zeros(len(freq)),np.power(np.absolute(F_f[0:freq_max]),2))
plt.xlabel(freq_label)
plt.ylabel(F_label)

# plot the complex frequency domain
plt.figure(3)
plt.clf()
plt.plot(freq, F_f.real[0:freq_max], 'bo', label='real')
plt.vlines(freq,np.zeros(len(freq)),F_f.real[0:freq_max])
plt.plot(freq, F_f.imag[0:freq_max], 'ro', label='imaginary')
plt.vlines(freq,np.zeros(len(freq)),F_f.imag[0:freq_max])
plt.xlabel(freq_label)
plt.ylabel('$Ff$')
plt.legend(loc='best')

# plot the signal in the time domain
plt.figure(4)
plt.clf()
plt.plot(t, f, label='$f$')
plt.plot(t, F_inv_F, 'r--', label='$F^{-1}Ff$')
plt.xlabel('$t$')
plt.ylabel('$f(t)$')
plt.legend()

# print the number of time samples to the screen
print('N = ', N)

###############################################################################

# DFT with only certain frequencies
freqs = np.array(specific_frequencies)
short_F_f = my_dft(f, freqs=freqs)

# create a full spectrum by adding zeros at the frequencies that weren't used
if N%2 == 0:
    full_F_f = np.zeros(N/2+1, dtype=complex)
else:
    full_F_f = np.zeros(int(np.ceil(N/2.0)), dtype=complex)

freq_index = 0
for i in range(int(np.floor(N/2))):
    if i == int(freqs[freq_index]):
        full_F_f[i] = short_F_f[freq_index]
        if freq_index < len(freqs)-1:
            freq_index += 1
    else:
        full_F_f[i] = 0.0
if N%2 == 0:
    full_F_f = np.concatenate((full_F_f, np.conj(np.flipud(full_F_f[1:-1]))))
else:
    full_F_f = np.concatenate((full_F_f, np.conj(np.flipud(full_F_f[1:]))))
    
# IDFT of the resulting spectrum
F_inv_full_F_f = np.fft.ifft(full_F_f)

# rescale freqs
if freq_axis == 'omega':
    freqs = np.divide(freqs, angular_frequency)
    
# plot the absolute value of the short frequency domain
plt.figure(5)
plt.clf()
plt.plot(freqs,np.power(np.absolute(short_F_f),2),'ko')
plt.vlines(freqs,np.zeros(len(freqs)),np.power(np.absolute(short_F_f),2))
plt.plot(0,0,'w.')
plt.xlabel(freq_label)
plt.ylabel(F_label)
plt.xlim(0,max(freqs)+min(freqs))

# plot the real and imaginary components of the new frequency domain
plt.figure(6)
plt.clf()
plt.plot(freqs, short_F_f.real[0:freq_max], 'bo', label='real')
plt.vlines(freqs,np.zeros(len(freqs)),short_F_f.real[0:freq_max])
plt.plot(freqs, short_F_f.imag[0:freq_max], 'ro', label='imaginary')
plt.vlines(freqs,np.zeros(len(freqs)),short_F_f.imag[0:freq_max])
plt.plot(np.linspace(0,max(freqs)+min(freqs)),np.zeros(50),'k--')
plt.xlabel(freq_label)
plt.ylabel('$Ff$')
plt.xlim(0,max(freqs)+min(freqs))
plt.legend(loc='best')

# plot the absolute value of the full frequency domain
plt.figure(7)
plt.clf()
plt.plot(freq,np.absolute(full_F_f[0:freq_max]),'ko')
plt.vlines(freq,np.zeros(len(freq)),np.absolute(full_F_f[0:freq_max]))
plt.xlabel(freq_label)
plt.ylabel(F_label)

# plot the reconstructed signal in the time domain
plt.figure(8)
plt.clf()
plt.plot(t, f, label='$f$')
plt.plot(t, F_inv_full_F_f, 'r--', label='$F^{-1}Ff$')
plt.xlabel('$t$')
plt.ylabel('$f(t)$')
plt.legend(loc='best')
