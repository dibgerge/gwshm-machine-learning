import pandas as pd
from packages import utkit
import numpy as np
from matplotlib import pyplot
from scipy.stats import pearsonr
from scipy.signal import get_window, hilbert


class RayModel(utkit.Signal):

    def __init__(self, period, sampling_rate, *args, **kwargs):
        """
        Parameters
        ----------
        period  : float
            The time period of the ray model in seconds.
            
        sampling_rate : float
            Sampling frequency in Hz.
        """
        n = int(period*sampling_rate)
        t = np.linspace(0, period, n)
        super().__init__(0, index=t, *args, **kwargs)

    def add(self, frequency, amplitude, pw, delay, window='hann'):
        """
        Add a ray to the model.
        
        Parameters
        ----------
        frequency : float
            The frequency of the ray to be added in Hz.
            
        amplitude : complex
            The amplitude of the ray. Can be a compelx quantity.
            
        pw : float
            The pulse width in seconds.
            
        delay : float
            The delay of the pulse (relative to 0, since signal always starts from time 0 seconds).
            
        window : string
            A string name for the window to be used. check scipy get_window for a complete list 
            of windows.
    
        Returns
        -------

        """
        pulse_time = self[delay:delay+pw].index.values
        win = amplitude * get_window(window, len(pulse_time))
        self[delay:delay+pw] += win * np.exp(1j*2*np.pi*frequency*(pulse_time - delay))

    def set(self, frequency, amplitude, pw, delay, window='hann'):
        self.loc[:] = 0
        pulse_time = self[delay:delay+pw].index.values
        win = amplitude * get_window(window, len(pulse_time))
        self[delay:delay+pw] = win * np.exp(1j*2*np.pi*frequency*(pulse_time - delay))

    @property
    def t(self):
        return self.index

