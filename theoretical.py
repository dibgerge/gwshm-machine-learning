""" Sum of windowed sinusoids is a single window???"""
import numpy as np
from matplotlib import pyplot
from packages import RayModel
from scipy.stats import uniform, norm
from scipy.signal import hilbert

pi = np.pi

pw = 40e-6
fs = 25e6
fc = 100e3

wave = RayModel(100e-6, fs)

for i in range(50):
    alpha = norm.rvs(1, scale=0.2) * np.exp(1j*uniform.rvs(loc=-pi, scale=2*pi))
    delay = uniform.rvs(scale=50e-6)
    wave.add(fc, alpha, pw, delay, window='hann')

wave = wave.window(index2=pw, win_fcn='hann')
fig, ax = pyplot.subplots(2, 1)
ax[0].plot(wave.real)

# find the window function
g = wave * np.exp(-1j*2*pi*fc*wave.index.values)

# find the fft value
Fwave = wave.fft(nfft=2**14)/np.sum(np.abs(g**2))
A = (Fwave.real(fc) + 1j*Fwave.imag(fc))

# reconstruct
wave_recon = g*np.exp(1j*2*pi*fc*wave.index.values)

ax[0].plot(wave_recon.real, '--')
ax[0].plot(abs(g))

#ax[1].plot(Fwave.abs())
ax[1].plot(g.fft().abs())
#ax[1].set_xlim([0, 200e3])
[axi.grid() for axi in ax]
pyplot.show()
