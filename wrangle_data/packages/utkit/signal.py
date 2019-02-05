import pandas as pd
import numpy as np
from scipy.fftpack import fft, fftfreq, fftshift, ifft
from scipy.signal import get_window, hilbert, fftconvolve, spectrogram, welch, coherence
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import simps
#from . import peakutils
from .signal2d import Signal2D
from matplotlib import pyplot
from scipy.special import jv
from scipy.optimize import curve_fit


class Signal(pd.Series):
    """
    Represents physical signals, by keeping track of the index (time/space) and corresponding
    values. This class extends the Pandas :class:`pandas.Series` class to make it more
    convenient to handle signals generally encountered in ultrasonics or RF.

    The class constructor is the same as that used for :class:`pandas.Series`. In addition,
    if :attr:`data` is an array, and :attr:`index` is a scalar, then :attr:`index` is interpreted as
    the sampling time, and the signal basis will be constructed as uniformly sampled at intervals
    specified by the scalar :attr:`index` and starting from 0.

    For example, to define a sine wave signal at 100 kHz and sampling interval of 1 microsecond,

    .. code-block:: python

        import numpy as np
        import utkit

        Ts = 1e-6
        t = np.arange(100)*Ts
        s = utkit.Signal(np.sin(2*np.pi*100e3*t), index=t)

    the last line which calls the constructor of :class:`Signal` is also equivalent to:

    .. code-block:: python

        s = utkit.Signal(np.sin(2*np.pi*100e3*t), index=Ts)

    The class :class:`Signal` provides various methods for signal reshaping, transforms,
    and feature extraction.
    """
    def __init__(self, data=None, index=None, *args, **kwargs):
        if index is not None and data is not None:
            if not hasattr(index, '__len__') and hasattr(data, '__len__'):
                index = np.arange(len(data)) * index
            elif not hasattr(index, '__len__') and not hasattr(data, '__len__'):
                index = [index]

        super().__init__(data, index, *args, **kwargs)
        # if not self.index.is_monotonic_increasing:
        #     raise ValueError('Index must be monotonically increasing.')

    @property
    def _constructor(self):
        """ Override the abstract class method so that all base class methods (i.e.
        methods in pd.Series) return objects of type utkit.Series when required.
        """
        return Signal

    def __hash__(self):
        return hash(str(self.values))

    def __call__(self, key=None, ts=None, **kwargs):
        """
        Make the Signal act like a function, where given index values are computed using an
        interpolant, and do not necessarily require to be at one of the current sample points.
        This method makes use of the SciPy interpolation method
        :class:`scipy.interpolate.InterpolatedUnivariateSpline`.

        Parameters
        ----------
        key : float, array_like
           The index value over which to compute the signal value. Can be either a scalar or a
           sequence of indices.

        ts : float, optional
            *ts* can be specified only if *key* has not been specified. This will re-sample the
            signal to the sample interval specified by *ts*.

        Returns
        -------
        value : float
           If `key` is a float, then the value of :class:`Signal` at given
           `key` is returned.

        value : Signal
           If `key` is a sequence, a new :class:`Signal` with its time base
           given by `key` is returned.

        Note
        ----
        This method take additional optional keyword arguments that are passed to the interpolant
        function. To see a description of supported arguments, refer to the documentation of
        :class:`scipy.interpolate.InterpolatedUnivariateSpline`.

        Note
        ----
        One difference in default values for the argument is that the extrapolation mode is set
        to 'zeros', instead of 'extrapolate'. This is based on the assumption that the signal
        goes to zero outside of its sampled range.
        """
        # print(ts)
        if key is None and ts is None:
            return self.copy()
        if key is not None and ts is not None:
            raise AttributeError("Only one of key or ts can be specified at one time, not both.")
        if ts is not None:
            key = np.arange(self.index[0], self.index[-1], ts)

        # set extrapolation to return 0 by default
        if 'ext' not in kwargs:
            kwargs['ext'] = 1

        self._interp_fnc = InterpolatedUnivariateSpline(self.index, self.values, **kwargs)
        if hasattr(key, '__len__'):
            return Signal(self._interp_fnc(key), index=key)
        else:
            return float(self._interp_fnc(key))

    def window(self, index1=None, index2=None, is_positional=False, win_fcn='boxcar',
               fftbins=False):
        """
        Applies a window to the signal within a given time range.

        Parameters
        ----------
        index1 : float or int, optional
            The start index/position of the window. Default value is minimum of index.

        index2 : float or int, optional
            The end index/position of the window. Default value is maximum of index.

        is_positional : bool, optional
            Indicates whether the inputs `index1` and `index2` are positional or value
            based. Default is :const:`False`, i.e. value based.

        win_fcn : string/float/tuple, optional
            The type of window to create. See the function
            :func:`scipy.signal.get_window()` for a complete list of
            available windows, and how to pass extra parameters for a
            specific window function.

        fftbins : bool, optional
            If True, then applies a symmetric window with respect to index of value 0.

        Returns
        -------
        Signal:
            The windowed Signal signal.

        Note
        ----
          If the window requires no parameters, then `win_fcn` can be a string.
          If the window requires parameters, then `win_fcn` must be a tuple
          with the first argument the string name of the window, and the next
          arguments the needed parameters. If `win_fcn` is a floating point
          number, it is interpreted as the beta parameter of the kaiser window.
        """
        wind = Signal(0, index=self.index)
        if is_positional:
            if isinstance(index1, float) or isinstance(index2, float):
                raise ValueError('Indices are floats, are you sure you want positional indices?')
            index1 = wind.index.values[index1]
            index2 = wind.index.values[index2]

        wind[index1:index2] = get_window(win_fcn, len(wind[index1:index2]))
        if fftbins:
            if wind[-index2:-index1].size == 0:
                raise IndexError('The signal does not have values at the negative of the indices '
                                 'supplied. Disable fftbins for one-sided windowing.')
            wind[-index2:-index1] = get_window(win_fcn, len(wind[-index2:-index1]))
        return self*wind

    def peaks(self, threshold=None, min_dist=None, by_envelop=False):
        """
        Finds the peaks by taking its first order difference. By using *thres* and
        *min_dist* parameters, it is possible to reduce the number of detected peaks.

        Parameters
        ----------
        threshold : float, [0., 1.]
            Normalized threshold. Only the peaks with amplitude higher than the
            threshold will be detected.

        min_dist : float
            The minimum distance in index units between ech detected peak. The peak with the highest
            amplitude is preferred to satisfy this constraint.

        by_envelop : bool
            Compute the peaks of the signal based on its envelop.

        Returns
        -------
        : ndarray
            Array containing the indexes of the peaks that were detected

        Notes
        -----
        This method is adapted from the peak detection method in
        [PeakUtils](http://pythonhosted.org/PeakUtils/)
        """
        y = self.operate('ne') if by_envelop else self.operate('n').abs()
        # pyplot.plot(y)

        if threshold is None:
            threshold = np.sqrt(y.energy()/len(self))

        if threshold > 1 or threshold <= 0:
            raise ValueError('Threshold should be in the range (0.0, 1.0].')

        if min_dist is None:
            min_dist = self.ts

        if min_dist <= 0.0:
            raise ValueError('min_dist should be a positive value.')
        # threshold = threshold * (y.max() - y.min()) + y.min()
        # find the peaks by using the first order difference
        dy = np.diff(y)
        peaks = np.where((np.hstack([dy, 0.]) < 0.)
                         & (np.hstack([0., dy]) > 0.)
                         & (y > threshold))[0]

        min_dist = int(min_dist/y.ts)
        if peaks.size > 1 and min_dist > 1:
            highest = peaks[np.argsort(y.iloc[peaks])][::-1]
            rem = np.ones(y.size, dtype=bool)
            rem[peaks] = False

            for peak in highest:
                if not rem[peak]:
                    sl = slice(max(0, peak - min_dist), peak + min_dist + 1)
                    rem[sl] = True
                    rem[peak] = False
            peaks = np.arange(y.size)[~rem]
        # pyplot.plot(y.iloc[peaks], 'or')
        # pyplot.show()
        return self.iloc[peaks]

    def normalize(self, option='max', inplace=False):
        """
        Normalizes the signal according to a given option.

        Parameters
        ----------
        option: string, optional
            Method to be used to normalize the signal. The possible options are:

            * *'max' (Default)* : Divide by the maximum of the signal, so that the normalized
                                  maximum has an amplitude of 1. :math:`\max \mathbf{s}`

            * *'energy'*: Divide by the signal energy: :math:`\sum_{i=1}^N s_i^2`

            * *'rms'* : The normalization is given by: :math:`\sqrt{\\frac{\sum_{i=1}^N s_i^2}{N}}`

        inplace : bool
            Change the Signal in place.
        Returns
        -------
        : Signal
            Signal with normalized amplitude.
        """
        if option == 'energy':
            fact = np.sqrt(self.energy())
        elif option == 'max':
            fact = np.max(np.abs(self.values))
        elif option == 'rms':
            fact = np.sqrt(self.energy()/len(self))
        else:
            raise ValueError('Unknown option value.')
        if inplace:
            self[:] = self/fact
            return self
        else:
            return self/fact

    def pad(self, extent, fill=0.0, position='split'):
        """
        Pad the signal with a given value to span a specified extent. If the extent is shorter
        than the current signal length, the signal will be truncated.

        Parameters
        ----------
        extent : float
            The desired signal length in *index* units.

        fill : {'edge', 'min', 'max', float}, optional
            Value used to fill the padded signal. If specified as 'edge', then the signal will be
            padded by the edge value of the signal. Thus padding to the right will pad with the
            value of the last sample, and padding to the left will use the value of the first
            sample.

        position : {'split', 'left', 'right'}
            How to fill the signal to span the required extent:
                * 'split': The signal will be padded evenly from left and right. If the number of
                  padded samples is odd, then right padding will be one sample more than left
                  padding.
                * 'left': Left (start) padding.
                * 'right': Right (end) padding.

        Returns
        -------
        : Signal
            A new padded signal.

        Note
        ----
            * If *extent* is not a multiple of the signal sampling rate, then the signal is
              padded to span the next possible value after the specified *extent*.
            * The original signal index will be kept intact, and no re-sampling will be made.
        """
        index_range = self.index.max() - self.index.min()
        npad = int(np.ceil((extent - index_range) / self.ts))
        npad_left, npad_right = npad, npad
        if position == 'right':
            npad_left = 0
        elif position == 'left':
            npad_right = 0
        elif position == 'split':
            npad_left, npad_right = int(np.floor(npad / 2)), int(np.ceil(npad / 2))
        else:
            raise ValueError('Unknown value for position.')

        if npad_right >= 0 and npad_left >= 0:
            index = np.concatenate((self.index[0] - np.arange(npad_left, 0, -1) * self.ts,
                                    self.index.values,
                                    self.index[-1] + np.arange(1, npad_right+1) * self.ts))
        else:
            index = self.index[-npad_left:npad_right]
        out = self.reindex(index)

        if fill == 'edge':
            return out.fillna(method='pad').fillna(method='bfill')
        elif fill == 'max':
            return out.fillna(self.max())
        elif fill == 'min':
            return out.fillna(self.min())
        else:
            return out.fillna(fill)

    def segment(self, threshold, pulse_width, min_dist=None, holdoff=None, win_fcn='boxcar'):
        """
        Segments the signal into a collection of signals, with each item in the collection,
        representing the signal within a given time window. This is usually useful to
        automate the extraction of multiple resolvable echoes.

        Parameters
        ----------
        threshold : float
            A threshold value (in dB). Search for echoes will be only for signal values
            above this given threshold. Note that the maximum threshold is 0 dB, since
            the signal will be normalized by its maximum before searching for echoes.

        pulse_width : float
            The expected pulse_width. This should have the same units as the units of the Signal
            index. If this is not known exactly, it is generally better to specify this
            parameter to be slightly larger than the actual pulse_width.

        min_dist : float
            The minimum distance between the peaks of the segmented signals.

        holdoff : float, optional
            The minimum index for which to extract a segment from the signal.

        win_fcn : string, array_like
            The window type that will be used to window each extracted segment (or echo). See
            :func:`scipy.signal.get_window()` for a complete list of available windows,
            and how to pass extra parameters for a specific window type, if needed.

        Returns
        -------
            : list
            A list with elements of type :class:`Signal`. Each Signal element represents an
            extracted segment.
        """
        if min_dist is None:
            min_dist = pulse_width

        pks = self.peaks(threshold, min_dist)

        if len(pks) == 0:
            return Signal2D(), Signal2D()

        if holdoff is not None:
            pks = pks[holdoff:]

        # remove segment if its end is over the limit of signal end
        if pks.index[-1] + pulse_width > self.index[-1]:
            pks = pks.iloc[:-1]

        out = Signal2D(0, index=self.index, columns=np.arange(len(pks)))
        lims = pd.DataFrame(0, index=['start', 'end', 'N'], columns=np.arange(len(pks)))

        for i, ind in enumerate(pks.index):
            win_st, win_end = ind-pulse_width, ind+pulse_width
            lims[i] = [win_st, win_end, len(self[win_st:win_end])]
            out[i] = self.window(index1=win_st, index2=win_end, win_fcn=win_fcn)

        return out, lims

    def operate(self, option='', norm_method='max', inplace=False):
        """
        This is a convenience function for common operations on signals. Returns the signal
        according to a given option.

        Parameters
        ----------
        options (string, char) :
            The possible options are (combined options are allowed)
            +--------------------+--------------------------------------+
            | *option*           | Meaning                              |
            +====================+======================================+
            | '' *(Default)*     | Return the raw signal                |
            +--------------------+--------------------------------------+
            | 'n'                | normalized signal by max amplitude   |
            +--------------------+--------------------------------------+
            | 'e'                | signal envelop                       |
            +--------------------+--------------------------------------+
            | 'd'                | decibel value                        |
            +--------------------+--------------------------------------+

        Returns
        -------
        : Signal
            The new signal with the specified operation.
        """
        n = len(self)
        yout = self
        if 'e' in option:
            # make hilbert transform faster by computing it at powers of 2
            n2 = 2 ** int(np.ceil(np.log2(n)))
            yout = Signal(abs(hilbert(yout.values, N=n2))[:n], index=self.index)
        if 'n' in option:
            yout = yout.normalize(norm_method)
        if 'd' in option:
            yout = 20*np.log10(abs(yout))
        if inplace:
            self[:] = yout
            return self
        else:
            return yout

    def fft(self, nfft=None, ssb=False):
        """
        Computes the Fast Fourier transform of the signal using :func:`scipy.fftpack.fft` function.
        The Fourier transform of a time series function is defined as:

        .. math::
           \mathcal{F}(y) ~=~ \int_{-\infty}^{\infty} y(t) e^{-2 \pi j f t}\,dt

        Parameters
        ----------
        nfft : int, optional
            Specify the number of points for the FFT. The default is the length of
            the time series signal.

        ssb : boolean, optional
            If true, returns only the single side band (components corresponding to positive
            frequency).

        Returns
        -------
        : Signal
            The FFT of the signal.
        """
        if nfft is None:
            nfft = self.size
        uf = Signal(fftshift(fft(self.values, n=nfft)), index=fftshift(fftfreq(nfft, self.ts)))
        return uf[uf.index >= 0] if ssb else uf

    def limits(self, thresh=None):
        """
        Computes the index limits where the signal first goes above a given threshold,
        and where it last goes below this threshold. Linear interpolation is used to find the
        exact point for this crossing.

        Parameters
        ----------
        thresh : float, optional, (0.0, 1.0]
            Normalized value where the signal first rises above and last falls below. If no value is
            specified, the default is the root mean square of the signal.

        Returns
        -------
        start_index, end_index (tuple (2,)):
            A two element tuple representing the *start_index* and *end_index* of the signal.
        """
        senv = self.operate('n')
        if thresh is None:
            thresh = self.std()
        if thresh <= 0 or thresh > 1:
            raise ValueError("`threshold` should be in the normalized (0.0, 1.0].")
        ind = np.where(senv.values >= thresh)[0]

        tout = []
        # make linear interpolation to get time value at threshold
        for i in [0, -1]:
            x1, x2 = self.index[ind[i]-1], self.index[ind[i]]
            y1, y2 = senv.iloc[ind[i]-1], senv.iloc[ind[i]]
            tout.append((thresh - y1) * (x2 - x1) / (y2 - y1) + x1)
        return tout[0], tout[1]

    def attenuation(self, a, d, pw, f=None, thres=None):
        """
        This methods computes the ultrasound attenuation of the signal. This is based on a piston
        transducer model.

        Parameters
        ----------
        a : float
            This is the diameter of the ultrasound transducer.

        d : float, optional
            Material thickness (propagation distance is assumed to be double the thickness)

        f : float, optional
            The frequency of the wave. If not provided, it is estimated from the signal.

        Returns
        -------

        References
        ----------
        Shmerr and Song 2007

        """
        echoes, _ = self.segment(threshold=thres, pulse_width=pw)
        if echoes.shape[1] < 2:
            ValueError('We need at least two echoes to compute the attenuation.')

        # compute the propagation distance for each echo
        x = 2*d*np.arange(1, echoes.shape[1]+1)

        # estimate the mean velocity
        c = np.mean(2*d/np.array([echoes[i].tof('corr', echoes[i-1]) for i in echoes.loc[:, 1:]]))

        # compute the FFT, which will be used for amplitudes computation
        Y = echoes.fft(ssb=True, axes=0).abs()

        # find the freuqnecies of the signal within the -6 dB bandwidth
        if f is None:
            # find the -6 dB limits of the frequency spectrum
            fmin, fmax = Y.iloc[:, 0].limits(0.5)
            amps = Y[fmin:fmax]
        elif f == 'max':
            fmin, fmax = Y.iloc[:, 0].limits(0.5)
            f = [(fmin+fmax)/2]
            amps = Y.apply(lambda v: Signal.__call__(v, f), axis=0)
        else:
            if not hasattr(f, '__len__'):
                f = [f]
            amps = Y.apply(lambda v: Signal.__call__(v, f), axis=0)

        def compute_att(xloc, alpha, A, flocal):
            p = 2*np.pi*flocal*a**2/(c * xloc)
            D = 1 - np.exp(1j*p)*(jv(0, p) - 1j*jv(1, p))
            return A * abs(D) * np.exp(-alpha * xloc)
            # return A*np.exp(-alpha * xloc)

        params = pd.DataFrame(0, index=amps.index, columns=['alpha', 'A'])
        for fi in amps.index:
            popt, pcov = curve_fit(lambda xp, alphap, ap: compute_att(xp, alphap, ap, fi), x,
                                   amps.loc[fi, :].values, p0=[0.02, 10])
            params.loc[fi] = popt
        return params

    def tof(self, method='corr', ref=None):
        """
        Computes the time of flight relative to another signal. Three different methods for
        computing the time of flight are currently supported.

        Parameters
        ----------
        method: string, optional
            The method to be used for computing the signal time of flight. The following methods
            are currently supported:

                * *corr* : Use a correlation peak to compute the time of flight relative to another
                  signal. Another signal should be provided as input for performing the correlation.

                * *max* : The maximum value of the signal is used to compute the time of flight.
                  This time of flight is relative to the signal's time 0.

                * *thresh* :  Compute the time the signal first crosses a given threshold value.
                The value of `ref` will be the required threshold, which should be normalized
                values between (0.0, 1.0], relative to the maximum value of the signal. If no
                reference value is given, the threshold is computed the root mean square of the
                signal.

        Returns
        -------
        : float
            The computed time of flight, with the same units as the Signal index.
        """
        if method.lower() == 'corr':
            if ref is None:
                raise ValueError('A reference signal should be specified to compute the tof using '
                                 'the correlation method.')
            c = fftconvolve(ref.values, self.values[::-1], mode='full')
            ind = self.size - np.argmax(c)
            return self.ts * ind
        elif method.lower() == 'max':
            return self.abs().idxmax()
        elif method.lower() == 'thresh':
            return self.limits(ref)[0]
        else:
            raise ValueError('method not supported. See documentation for supported methods.')

    def spectrogram(self, width, overlap=0, **kwargs):
        """
        Computes the spectrogram (short-time Fourier transform) of the signal. This method uses
        the function :func:`scipy.signal.spectrogram`.

        Parameters
        ----------
        width : float
            Substitute for the argument *nperseg* in the function :func:`scipy.signal.spectrogram`.
            Here, *width* has same units as *index*.

        overlap : float, optional
            Substitute for the argument *noverlap* in the function :func:`scipy.signal.spectrogram`.
            Here, *overlap was same units as *index*.

        Returns
        -------
        : Signal2D
            A Signal2D class representing the

        Note
        ----
        For other supported keyword arguments, see the documentation of
        :func:`scipy.signal.spectrogram`. However, the two arguments *nperseg* and *noverlap*
        should not be used.
        """
        nperseg = int(width * self.fs)
        nol = int(overlap * self.fs)
        f, t, S = spectrogram(self.values, fs=self.fs, nperseg=nperseg, noverlap=nol, **kwargs)
        return Signal2D(S, index=f, columns=t)

    def psd(self, width, overlap=0, **kwargs):
        """
        Computes the periodogram of the signal. This method uses the function
        :func:`scipy.signal.welch`.

        Parameters
        ----------
        width : float
            Substitute for the argument *nperseg* in the function :func:`scipy.signal.spectrogram`.
            Here, *width* has same units as *index*.

        overlap : float, optional
            Substitute for the argument *noverlap* in the function :func:`scipy.signal.spectrogram`.
            Here, *overlap was same units as *index*.

        Returns
        -------
        : Signal2D
            A Signal2D class representing the

        Note
        ----
        For other supported keyword arguments, see the documentation of
        :func:`scipy.signal.welch`. However, the two arguments *nperseg* and *noverlap*
        should not be used.
        """
        nperseg = int(width * self.fs)
        nol = int(overlap * self.fs)
        f, pxx = welch(self.values, fs=self.fs, nperseg=nperseg, noverlap=nol, **kwargs)
        return Signal(pxx, index=f)

    def sparse_pse(self, threshold, fc, pulse_width, overlap=0, nfft=None, win_fcn='boxcar'):
        """
        Computes the sparse power spectral estimate
        Parameters
        ----------
        fc
        width
        overlap
        nfft

        Returns
        -------

        """
        echoes, lims = self.segment(threshold=threshold, pulse_width=pulse_width,
                                    min_dist=pulse_width-overlap, win_fcn=win_fcn)
        if echoes.shape[1] == 0:
            return Signal(0, index=[0])
        Y = echoes.fft(shape=nfft, axes=0, ssb=True).abs()**2
        return Y(index=fc)

    def pse(self, fc, pulse_width, overlap=0, nfft=None, win_fcn='boxcar'):
        """
        Computes the continuous power spectral estimate.

        Parameters
        ----------
        fc
        pulse_width
        overlap
        nfft
        win_fcn

        Returns
        -------

        """
        if overlap >= pulse_width:
            raise ValueError('overlap should be smaller than pulse_width.')

        delay = 0
        a = 0
        nbins = 0
        while delay + pulse_width <= self.index[-1]:
            s = self.window(index1=delay, index2=delay+pulse_width, win_fcn=win_fcn)
            Y = s.fft(ssb=True, nfft=nfft).abs()
            a += Y(fc)**2
            nbins += 1
            delay += pulse_width-overlap

        return a, nbins

    def coherence(self, other, width, overlap=0, **kwargs):
        """
        Compute the short-time correlation coefficient of the signal. Uses the function
        :func:`scipy.signal.coherence` for this computation.

        Parameters
        ----------
        other : Signal
            The other Signal that will be used to perform the short time cross correlation.

        width : float
            Window size (in Signal index units) that will be used in computing the short-time
            correlation coefficient.

        overlap : float, optional
            Units (index units) of overlap between consecutive computations.

        Returns
        -------
        : array_like
            The computed short tiem cross-correlation function.
        """
        other = other(self.index)
        nperseg = int(width * self.fs)
        nol = int(overlap * self.fs)
        f, cxy = coherence(self.values, other.values, self.fs, nperseg=nperseg, noverlap=nol,
                           **kwargs)
        return Signal(cxy, index=f)

    def filter_freq(self, cutoff, option='lp', win_fcn='boxcar'):
        """
        Applies a frequency domain filter to the signal.

        Parameters
        ----------
        cutoff : float or (2,) tuple
            The cutoff frequency (Hz) of the filter. This is a scalar value if type
            is ``'lp'`` or ``'hp'``. When type is ``'bp'``, cutoff  should be a 2 element list,
            where the first element specifies the lower cutoff frequency, and the second element
            specifies the upper cutoff frequency.

        option : string, optional
            The type of filter to be used.

            +--------------------+-----------------------------------------+
            | *option*           | Meaning                                 |
            +====================+=========================================+
            | 'lp' *(Default)*   | Low-pass filter                         |
            +--------------------+-----------------------------------------+
            | 'hp'               | High-pass filter                        |
            +--------------------+-----------------------------------------+
            | 'bp'               | Band-pass filter                        |
            +--------------------+-----------------------------------------+

        win_fcn : string, optional
            Apply a specific window in the frequency domain. See the function
            :func:`scipy.signal.get_window` for a complete list of available windows, and how to
            pass extra parameters for a specific window function.

        Returns
        -------
        : Signal
            The filtered Signal.
        """
        fdomain = self.fft()
        index1 = 0
        index2 = self.fs / 2.0
        if option == 'lp':
            index2 = cutoff
        elif option == 'hp':
            index1 = cutoff
        elif option == 'bp':
            index1 = cutoff[0]
            index2 = cutoff[1]
        else:
            raise ValueError('The value for type is not recognized.')

        fdomain = fdomain.window(index1=index1, index2=index2, win_fcn=win_fcn, fftbins=True)
        return Signal(np.real(ifft(fftshift(fdomain))), index=self.index)

    def frequency(self, option='center', threshold=None, nfft=None):
        """
        Computes the center or peak frequency of the signal.

        Parameters
        ----------
        option : str, {'center', 'peak'}
            Specify whether to compute the center or peak frequency of the signal.

        threshold : float, optional, (0.0, 1.0]
            Threshold value indicating the noise floor level. Default value is the root mean
            square.

        nfft : bool, optional
            Since this computation is based on the Fourier transform, indicate the number of
            points to be used on computing the FFT.

        Returns
        -------
        : float
            The value of the center frequency.
        """
        if (threshold is not None) and (threshold <= 0 or threshold > 1):
            raise ValueError("Threshold should be in the range (0.0, 1.0].")
        fdomain = self.fft(ssb=True, nfft=nfft).abs()

        if option == 'center':
            yn = fdomain.operate('n')
            minf, maxf = yn.limits()
            return (minf+maxf)/2.0
        if option == 'peak':
            return fdomain.idxmax()
        raise ValueError('`option` value given is unknown. Supported options: {"center", "peak"}.')

    def bandwidth(self, threshold=None, nfft=None):
        """
        Computes the bandwidth of the signal by finding the range of frequencies
        where the signal is above a given threshold.

        Parameters
        ----------
        threshold : float, optional, (0.0, 1.0]
            The normalized threshold for which to compute the bandwidth. If this is not
            specified, the threshold is set to the root mean square value of the signal.

        nfft : bool, optional
            Since this computation is based on the Fourier transform, indicate the number of
            points to be used on computing the FFT.

        Returns
        -------
        : float
            The total signal bandwidth.
        """
        if threshold <= 0 or threshold > 1:
            raise ValueError("Threshold should be in the range (0.0, 1.0].")
        fdomain = self.fft(ssb=True, nfft=nfft).abs()
        lims = fdomain.limits(threshold)
        return lims[1] - lims[0]

    def maxof(self, option='peak'):
        """
        Computes the maximum of the signal according to a given method.

        Parameters
        ----------
        option : str, optional
            The method to be used to compute the maximum. Supported options are:

            ==================    ======================================
            *option*               Meaning
            ==================    ======================================
            abs                   Max of the signal absolute value
            env                   Max of the signal envelop
            peak *(Default)*      Max of the raw signal
            fft                   Max of the signal FFT magnitude
            ==================    ======================================

        Returns
        -------
        : float
            The maximum value of the specified signal form.
        """
        if option == 'peak':
            y = self
        elif option == 'abs':
            y = self.abs()
        elif option == 'env':
            y = self('e')
        elif option == 'fft':
            yf = self.fft(ssb=True)
            y = yf.abs()/yf.size
        else:
            raise ValueError("The value for option is unknown. Should be: 'abs',"
                             "'env', 'peak', or 'fft'.")
        return np.max(y.values)

    def energy(self):
        """
        Computes the energy of the given waveform in the specified domain.

        Returns
        -------
        : float
            The computes energy in the signal.
        """
        return np.power(self, 2).sum()

    def remove_mean(self):
        """
        Subtracts the mean of the signal.

        Returns
        -------
        : Signal
            The :class:`Signal` with its mean subtracted.
        """
        return self - self.mean()

    @property
    def ts(self):
        """
        Get the signal sampling period.

        Returns
        -------
        : float
        """
        return np.mean(np.diff(self.index))

    @property
    def extent(self):
        """ Get the signal index extent. """
        return self.index.max() - self.index.min()

    @property
    def fs(self):
        """ Get the signal sampling frequency. """
        return 1.0/self.ts

    @property
    def real(self):
        return Signal(np.real(self), index=self.index)

    @property
    def imag(self):
        return Signal(np.imag(self), index=self.index)