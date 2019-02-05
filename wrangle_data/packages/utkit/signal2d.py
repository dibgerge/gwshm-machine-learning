import pandas as pd
import numpy as np
from scipy.signal import hilbert, get_window, fftconvolve
from scipy.interpolate import griddata
from scipy.fftpack import fft2, fftfreq, fftshift, ifft2
from .signal3d import Signal3D
from matplotlib import pyplot
from scipy.special import jv
from scipy.optimize import curve_fit


class Signal2D(pd.DataFrame):
    """
    Extends :class:`pandas.DataFrame` class to support operations commonly required
    in radio frequency signals, and especially in ultrasonics. The axis convention used is:

        * **Axis 1 (columns)**: *X*-direction
        * **Axis 0 (index)**: *Y*-direction

    For example, in the context of ultrasonic inspection, the *X*-direction would represent
    the spatial line scan, and the *Y*-direction represents the signal time base, which can
    be scaled to represent the ultrasonic beam depth through the material.

    The class constructor is similar as that of :class:`pandas.DataFrame` with the added
    option of specifying only the sampling intervals along the *X* and *Y* directions.
    Thus, if *index* and/or *columns* are scalars, which *data* is a 2-D array, then
    the Signal2D basis are constructed starting from 0 at the given sampling intervals.

    If data input is a dictionary, usual rules from :class:`pandas.DataFrame` apply, but index
    can still be a scalar specifying the sampling interval.
    """
    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False):
        if index is not None and data is not None:
            if not hasattr(index, '__len__'):
                if isinstance(data, dict):
                    if hasattr(data[list(data.keys())[0]], '__len__'):
                        datalen = len(data[list(data.keys())[0]])
                    else:
                        datalen = 0
                elif hasattr(data, '__len__'):
                    datalen = len(data)
                else:
                    datalen = 0
                if datalen > 0:
                    index = np.arange(datalen) * index

        if columns is not None and data is not None:
            if not hasattr(columns, '__len__'):
                if isinstance(data, dict):
                    datalen = 0
                elif isinstance(data, pd.Series):
                    datalen = 0
                elif isinstance(data, pd.DataFrame):
                    datalen = data.shape[1]
                elif hasattr(data, '__len__'):
                    datalen = len(data[0])
                else:
                    datalen = 0
                if datalen > 0:
                    columns = np.arange(datalen) * columns

        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy)
        # check for axes monotonicity
        if not self.index.is_monotonic_increasing:
            raise ValueError('Index must be monotonically increasing.')
        # if not self.columns.is_monotonic_increasing:
        #     raise ValueError('Columns must be monotonically increasing.')

    @property
    def _constructor(self):
        return Signal2D

    @property
    def _constructor_sliced(self):
        from .signal import Signal
        return Signal

    @property
    def _constructor_expanddim(self):
        return Signal3D

    def _get_axes_numbers(self, axes):
        """
        Returns the numbers of axes, given a list of them.

        Parameters
        ----------
        axes : int, str, list
            The axes list to be evaluated. axes could be named alias for corresponding axes
            numbers.

        Returns
        -------
        : list
            Axes as numbers. If axes was None, all available axes are returned. If axes was a
            scalar, return a list of size one.
        """
        if axes is None:
            return [0, 1]

        if isinstance(axes, str):
            return [self._get_axis_number(axes)]
        elif hasattr(axes, '__len__'):
            return [self._get_axis_number(ax) for ax in axes]
        return [axes]

    def __call__(self, index=None, columns=None, **kwargs):
        """
        Interpolate the axes. This function used :func:`scipy.interpolate.griddata`.

        Parameters
        ----------
        index : array_like
            New index values to compute the interpolation.

        columns : array_like
            New columns values to compute the interpolation.

        Returns
        -------
        : Signal2D
            A new Signal2D object computed at the new given axes values.

        Notes
        -----
        Other keyword arguments are passed directly to the interpolation function
        :func:`scipy.interpolate.griddata`.
        """
        if index is None and columns is not None:
            from . import Signal
            return self.apply(lambda x: Signal.__call__(x, columns), axis=1)
        if columns is None and index is not None:
            from . import Signal
            return self.apply(lambda x: Signal.__call__(x, index), axis=0)

        index, columns = np.array(index), np.array(columns)
        if index.ndim == 0:
            index = np.array([index])
        if columns.ndim == 0:
            columns = np.array([columns])
        if (index.ndim != 1) or (columns.ndim != 1):
            raise TypeError('New index and columns must be one dimensional arrays.')

        xg, yg = np.meshgrid(self.columns, self.index, indexing='xy')
        new_xg, new_yg = np.meshgrid(columns, index, indexing='xy')
        vals = griddata((yg.ravel(), xg.ravel()), self.values.ravel(), (new_yg, new_xg), **kwargs)
        return Signal2D(vals, index=index, columns=columns)

    def reset_rate(self, ts, axes=None, **kwargs):
        """
        Re-samples the Signal to be of a specified sampling rate. The method uses interpolation
        to recompute the signal values at the new samples.

        Parameters
        ----------
        ts : float, array_like (2,)
            The sampling rate for the axes specified. If it is a scalar, and axes is
            :const:`None`, the same sampling rate will be set for both axes.

        axes : {0/'y'/'index', 1/'x'/'columns'}, optional
            The axes along which to apply the re-sampling. If it is not set, both axes will be
            re-sampled.

        Returns
        -------
        : Signal2D
            A copy of the Signal2D with the new sampling rate.
        """
        ts = self._set_val_on_axes(ts, axes, self.ts)
        indices = [np.arange(self.axes[i][0], self.axes[i][-1], ts[i]) for i in range(2)]
        return self(index=indices[0], columns=indices[1], **kwargs)

    def operate(self, option='', norm='max', axis=0):
        """
        Returns the signal according to a given option.

        Parameters
        ----------
        option : string/char, optional
            The possible options are (combined options are allowed):

             +--------------------+--------------------------------------+
             | *option*           | Meaning                              |
             +====================+======================================+
             | '' *(Default)*     | Return the raw signal                |
             +--------------------+--------------------------------------+
             | 'n'                | normalized signal                    |
             +--------------------+--------------------------------------+
             | 'd'                | decibel value                        |
             +--------------------+--------------------------------------+

        axis : int, optional
            Only used in the case option specified 'e' for envelop. Specifies along which axis to
            compute the envelop.

        Returns
        -------
        : Signal2D
            The modified Signal2D.
        """
        yout = self
        if 'e' in option:
            # make hilbert transform faster by computing it at powers of 2
            n = self.shape[axis]
            n = 2**int(np.ceil(np.log2(n)))
            yout = np.abs(hilbert(yout.values, N=n, axis=axis))
            yout = yout[:self.shape[0], :self.shape[1]]
        if 'n' in option:
            # TODO: call normalize funciton
            yout = yout/np.abs(yout).max().max()
        if 'd' in option:
            yout = 20*np.log10(np.abs(yout))
        return Signal2D(yout, index=self.index, columns=self.columns)

    def _cook_args(self, val, axes, default=None):
        """
        Changes size of val based on given axes.

        Parameters
        ---------
        val : float, array_like
            This is the value to be tested

        axes : array_like
            The axis for which the *val* corresponds.

        default : list
            Default values over all axes.

        Returns
        -------
        : numpy.ndarray
            A 2-element array representing the filled *val* with missing axis value.
        """
        if val is None:
            return default

        # if val is scalar, make it same size as given axes
        if not hasattr(val, '__len__'):
            val = np.ones(len(axes))*val

        # if val is an array, make sure it has same size as axes
        if len(val) != len(axes):
            raise ValueError('value and axes have different lengths.')

        # if defaults given, fill up all dimensions not specified by axes to default values
        if default is not None:
            default = np.array(default)
            default[axes] = val
            return default
        return np.array(val)

    def fft(self, ssb=False, axes=None, shape=None, **kwargs):
        """
        Computes the Fourier transform in two dimensions, or along a specified axis.

        Parameters
        ----------
        ssb : bool, optional
            Determines if only the single sided Fourier transform will be returned.

        axes : int, array_like, optional
            The axes along which to compute the FFT.

        shape : int, array_like, optional
            The size of the fft

        Returns
        -------
        : Signal2D
            A new signal representing the Fourier transform.

        Note
        ----
        Keyword arguments can be given to the the underlying Fourier transform function
        :func:`scipy.fftpack.fft2`.
        """
        axes = self._get_axes_numbers(axes)
        if shape == 'next':
            shape = 2 ** np.ceil(np.log2(self.shape))
            shape = shape[axes[0]] if len(axes) == 1 else shape

        shape = self._cook_args(shape, axes)
        if shape is not None:
            shape = shape.astype(int)

        fval = fftshift(fft2(self.values, axes=axes, shape=shape, **kwargs), axes=axes)

        coords = [self.index, self.columns]
        for ax in axes:
            coords[ax] = fftshift(fftfreq(fval.shape[ax], self.ts[ax]))
        s = Signal2D(fval, index=coords[0], columns=coords[1])

        if ssb:
            for ax in axes:
                coords[ax] = coords[ax][coords[ax] >= 0]
            s = Signal2D(s, index=coords[0], columns=coords[1])
        return s

    def window(self, index1=None, index2=None, axes=None, win_fcn='boxcar'):
        """
        Applies a window to the signal within a given time range.

        Parameters
        ----------
        index1 : {float, int, array_like}, optional
            The start index/position of the window. Default value is minimum of index and columns.
            If *index1* is a two_element array, then it specifies the start positions for both axes.

        index2 : {float, int, array_like}, optional
            The end index/position of the window. Default value is maximum of index and columns.
            If *index2* is a two_element array, then it specifies the end positions for both axes.

        axes : {int, string, array_like}, optional
            The axes names/numbers along which to apply the window.

        win_fcn : string/float/tuple, optional
            The type of window to create. See the function :func:`scipy.signal.get_window()` for
            a complete list of available windows, and how to pass extra parameters for a
            specific window function.

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
        axes = self._get_axes_numbers(axes)
        index1 = self._cook_args(index1, axes)
        index2 = self._cook_args(index2, axes)

        if index1 is None:
            index1 = [self.axes[i][0] for i in axes]
        if index2 is None:
            index2 = [self.axes[i][-1] for i in axes]

        win2d = Signal2D(0, index=self.index, columns=self.columns)

        win = []
        for ax in sorted(axes):
            st, en = self.axes[ax].slice_locs(index1[ax], index2[ax])
            win.append(get_window(win_fcn, en - st, fftbins=False))

        # Case where we want to window the two dimensions
        if len(win) == 2:
            win = np.sqrt(np.outer(win[0], win[1]))
            win2d.loc[index1[0]:index2[0], index1[1]:index2[1]] = win
        elif len(win) == 1:
            # aply the same window across the other axis
            win = np.repeat(win[0][:, np.newaxis], self.shape[axes[0]-1], axes[0]-1)
            win_slice = win2d.select(lambda x: (x >= index1[0]) & (x <= index2[0]), axis=axes[0])
            win_slice.loc[:] = win
            win2d.update(win_slice)
        else:
            raise ValueError('Could not make window.')
        return self*win2d

    def filter_freq(self, low_freq=None, high_freq=None, axes=None, win_fcn='boxcar'):
        """
        Applies a filter in the frequency domain.

        Parameters
        ----------
        low_freq : scalar, array_like, optional
            The lower cutoff frequency for the filter. All frequencies less than this will be
            filtered out. If this is scalar, and *axes* is :const:`None`, then the same
            low_frequency will be applied for both axes.

        high_freq : scalar, array_like, optional
            The upper cutoff frequency for the filter. All frequencies higher than this will be
            filtered out. If this is scalar, and *axes* is :const:`None`, then the same
            high_frequency will be applied for both axes.

        axes : {int, string}, optional
            The axes along which to filter the 2D signal.

        win_fcn : {string, tuple}, optional
            The window type to apply for performing the filtering in the frequency domain. See the
            function :func:`scipy.signal.get_window()` for a complete list of available windows,
            and how to pass extra parameters for a specific window function.

        Returns
        -------
        : Signal2D
            The new filtered signal.
        """
        axes = self._get_axes_numbers(axes)
        fdomain = self.fft(axes=axes)
        low_freq = self._cook_args(low_freq, axes)
        high_freq = self._cook_args(high_freq, axes)

        if low_freq is None:
            low_freq = [0]*len(axes)
        if high_freq is None:
            high_freq = [self.ts[ax]/2. for ax in axes]

        fupper, flower = fdomain.copy(), fdomain.copy()
        for ax in axes:
            fupper = fupper.select(lambda x: x >= 0, axis=ax)
            flower = flower.select(lambda x: x < 0, axis=ax)

        fupper = fupper.window(index1=low_freq, index2=high_freq, axes=axes, win_fcn=win_fcn)
        flower = flower.window(index1=-np.array(high_freq), index2=-np.array(low_freq),
                               axes=axes, win_fcn=win_fcn)
        fdomain.update(fupper)
        fdomain.update(flower)
        vals = fftshift(fdomain.values, axes=axes)
        ift = ifft2(vals, axes=axes, shape=np.array(self.shape)[axes])
        return Signal2D(np.real(ift), index=self.index, columns=self.columns)

    # def periodogram(self, pulse_width, axes=0, fc='max', overlap=0, **kwargs):
    #     """
    #     Currently only along the 0 axis.
    #     Parameters
    #     ----------
    #     fc
    #     width
    #     overlap
    #     nfft
    #
    #     Returns
    #     -------
    #
    #     """
    #     start = self.index[0]
    #     amp = 0
    #     while start < self.index[-1]:
    #         s_seg = self.loc[start:(start + width)]
    #         if len(s_seg) <= 1:
    #             break
    #         n = len(s_seg)
    #         yf = s_seg.fft(ssb=True, axes=0, **kwargs).abs()
    #         start += width - overlap
    #         amp += 2 * yf(fc).values**2 / n
    #     return amp

    def shift_axes(self, shift, axes=None):
        """
        Shifts an axis (or both axis) by a specified amount.

        Parameters
        ----------
        shift : float, array_like
            The amount to shift the axis. If *shift* is an array, it should have the same size as
            *axes*, if specified.

        axes : int, string, array_like, optional
            The axes to shift. If not specified, all axes will be shifted by the given *shift*
            value.

        Returns
        -------
        : Signal2D
            A new Signal2D with shifted axes.
        """
        axes = self._get_axes_numbers(axes)
        shift = self._cook_args(shift, axes, [0.0, 0.0])
        return Signal2D(self.values, index=self.index-shift[0], columns=self.columns-shift[1])

    def scale_axes(self, scale, axes=None):
        """
        Scales a given axis (or both) by a given amount.

        Parameters
        ----------
        scale : float, array_like
            The amount to scale the axis. If *scale* is an array, it should have the same size as
            *axes*, if specified.

        axes : int/string, optional
            The axes to scale. If not specified, all axes will be scaled by the given *scale* value.
        Returns
        -------
        : Signal2D
            A copy of Signal2D with scaled axes.
        """
        axes = self._get_axes_numbers(axes)
        scale = self._cook_args(scale, axes, [1., 1.])
        return Signal2D(self.values, index=self.index*scale[0], columns=self.columns*scale[1])

    def skew(self, angle, axes=1, interpolate=False, ts=None, **kwargs):
        """
        Applies a skew transformation on the data.

        Parameters
        ----------
        angle : float, array_like
            The angle to skew the Scan2D coordinates. If *angle* is an array, it should
            have the same size as *axes*. If it is a scalar, the angle will be applied to all
            specified *axes*

        axes : integer, str, optional
            The axis along which to skew. If *axes* is set to :const:`None`, then both axes are
            skewed.

        interpolate : bool, optional
            If const:`True`, realign the skewed axes on a regular grid, and use interpolation to
            recompute the values of the Signal2D at the new regular grid. The new grid will be
            computed to span the new range of the axes.

            Otherwise, no interpolation will be formed, however the grid will be similar to the
            current object grid. That is, the skewed grid coordinates are reset to match the
            nearest neighbor from the original grid.

        ts : float, array_like, optional
            Only required if *interpolate* is set to :const:`True`. Specifies the sampling
            interval for the new regular grid used in the interpolation. If not specified,
            then by default, the current sampling intervals of the Signal2D object will be used.

        Returns
        -------
        signal2D : Signal2D
            If *interpolate* is :const:`True`, then a new Signal2D object is returned,
            after interpolation onto a regular grid.

        X, Y, val : tuple
            If *interpolate* is :const:`False`, then a tuple is returned, composed of the the
            new coordinates (X, Y), where now X and Y are 2-D matrices with same shape as the
            dataframe. The values are also returned for convenience, but they are the same as the
            current Signal2D object.
        """
        angle = self._cook_args(angle, self._get_axes_numbers(axes), [0, 0])

        tan_angle = np.tan(np.deg2rad(angle))
        skew_matrix = np.array([[1.0, tan_angle[1]], [tan_angle[0], 1.0]])

        x, y = np.meshgrid(self.columns, self.index, indexing='xy')
        xskew, yskew = np.dot(skew_matrix, np.array([x.ravel(), y.ravel()]))
        x, y = xskew.reshape(x.shape), yskew.reshape(y.shape)

        if interpolate:
            ts = self._cook_args(ts, axes, self.ts)
            xnew = np.arange(np.min(x), np.max(x) + ts[1], ts[1])
            ynew = np.arange(np.min(y), np.max(y) + ts[0], ts[0])
            xv, yv = np.meshgrid(xnew, ynew, indexing='xy')
            vals = griddata((y.ravel(), x.ravel()), self.values.ravel(), (yv, xv), **kwargs)

            if ('method' in kwargs) and (kwargs['method'] == 'nearest'):
                oldlabels, labels, newlabels = [self.x, self.y], [y, x], [ynew, xnew]

                # The following removes the effect at the edges, where the edge values are
                # extruded into the region outside the frame.
                for ax in axes:
                    min_val, max_val = np.min(labels[ax], axis=ax), np.max(labels[ax], axis=ax)
                    smin = pd.Series(min_val, index=oldlabels[ax]).reindex(ynew, method='nearest')
                    smax = pd.Series(max_val, index=oldlabels[ax]).reindex(ynew, method='nearest')
                    vals[newlabels[ax] - smin.values.reshape(-1, 1) < 0] = np.nan
                    vals[newlabels[ax] - smax.values.reshape(-1, 1) > 0] = np.nan
            return Signal2D(vals, index=ynew, columns=xnew)
        else:
            xv = np.round(x / self.ts[1]) * self.ts[1]
            yv = np.round(y / self.ts[0]) * self.ts[0]
            return xv, yv, self.values

    def pad(self, extent, axes=None, fill=0.0, position='split'):
        """
        Adds padding along the given axes.

        Parameters
        ----------
        extent : scalar, 2-element array
            The desired extent of the axis that requires padding. If the given extent is smaller
            than the current axis extent, the signal2D will be truncated.

        axes : {0/'y'/index or 1/'x'/'columns, None}
            The axes along which to apply the padding. If :const:`None` is specified, then the
            Signal2D will be padded along both axes.

        fill : {'min', 'max', scalar}, optional
            The value to fill the padded regions:
                * 'min': Pad with values of the minimum amplitude in the signal.
                * 'max': Pad with the value of the maximum amplitude in the signal.
                * scalar: otherwise, a custom scalar value can be specified for the padding.

        position : {'start', 'end', 'split'}
            How to apply the padding to the Signal2D:
                * 'start' : apply the padding at the beginning of the axes. i.e., to the left for
                  the columns, and to the top for the index.
                * 'end' : apply the padding at the end of the axes.
                * 'split': split the padding to be half at the start and half at the end. If the
                  number of samples padded is odd, then the end will have one more sample padded
                  than the start.

        Returns
        -------
        : Signal2D
            A new Signal2D object will the axes padded. The sampling interval of the padded
            region is equal to the mean sampling rate of the corresponding axis.

        """
        extent = self._set_val_on_axes(extent, axes, np.array(self.extent) + np.array(self.ts))
        npad = np.array([int(np.ceil(extent[i]/self.ts[i] - self.shape[i])) for i in [0, 1]])
        npad_start, npad_end = npad, npad
        if position == 'end':
            npad_start = [0, 0]
        elif position == 'start':
            npad_end = [0, 0]
        elif position == 'split':
            npad_start, npad_end = np.floor(npad / 2), np.ceil(npad / 2)
        else:
            raise ValueError('Unknown value for position.')
        ax = []
        for i in [0, 1]:
            if npad_end[i] >= 0 and npad_start[i] >= 0:
                left_part = self.axes[i].values[0] - np.arange(npad_start[i], 0, -1) * self.ts[i]
                right_part = self.axes[i].values[-1] + np.arange(1, npad_end[i]+1) * self.ts[i]
                ax.append(np.concatenate((left_part, self.axes[i].values, right_part)))
            else:
                ax.append(self.axes[i][-npad_start[i]:npad_end[i]])
        if fill == 'min':
            fill = self.min().min()
        elif fill == 'max':
            fill = self.max().max()
        return self.reindex(index=ax[0], columns=ax[1], fill_value=fill)

    def pad_coords(self, index=None, columns=None, fill=0.0):
        """
        This allows padding by specifying the start and end of the coordinates for each of the axes.

        Parameters
        ----------
        index : 2-element array, optional
            Specifies the start and end of the index.

        columns : 2-element array, optional
            Specified the start and end of the columns.

        fill : float, optional
             The value to fill the padded regions:
                * 'min': Pad with values of the minimum amplitude in the signal.
                * 'max': Pad with the value of the maximum amplitude in the signal.
                * scalar: otherwise, a custom scalar value can be specified for the padding.

        Returns
        -------
        : Signal2D
            The padded signal.
        """
        if index is not None and len(index) != 2:
            raise ValueError('index_range should have a size of 2.')

        if columns is not None and len(columns) != 2:
            raise ValueError('columns_range should have a size of 2.')

        x, y = self.columns, self.index
        if index is not None:
            # find values less than x
            y = np.arange(index[0], index[1], self.ts[0])

        if columns is not None:
            x = np.arange(columns[0], columns[1], self.ts[1])

        out = self.reindex(index=y, columns=x, method='nearest', fill_value=fill)
        # out.loc[out.index < self.index[0]] = fill
        # out.loc[out.index > self.index[-1]] = fill
        # out.loc[:, out.columns < self.columns[0]] = fill
        # out.loc[:, out.columns > self.columns[-1]] = fill
        return out

    def flip(self, axes=None):
        """
        Flips the values without flipping corresponding X/Y-axes coordinates.

        Parameters
        ----------
        axes : int/string, optional
            The axis along which to flip the values. axis can be 0/'y'/'index'. or 1/'x'/'columns'.
            If axis is set to :const:`None`, both axes will be flipped.

        Returns
        --------
        : Signal2D
            A copy of Signal2D with axis flipped.
        """
        axes = self._make_axes_as_num(axes)
        vals = self.values
        if 0 in axes:
            vals = vals[::-1, :]
        if 1 in axes:
            vals = vals[:, ::-1]
        return Signal2D(vals, index=self.index, columns=self.columns)

    def roll(self, value, axes=None):
        """
        Circular shift by a given value, along a given axis.

        Parameters
        ----------
        value : float
            The amount (in X-Y coordinates units) to shift.

        axes : string/int, optional
            The axis along which to shift. Options are 0/'Y'/'index' or 1/'X'/columns. By default,
            the uFrame is flattened before shifting, after which the original shape is restored.
            See numpy.roll for more information.

        Returns
        -------
        : Signal2D
            A copy of Signal2D after applying the circular shift.
        """
        axes = self._make_axes_as_num(axes)
        value = self._set_val_on_axes(value, axes, [0., 0.])
        out_val = self.values
        for ax in axes:
            indexes = int(np.around(value/self.ts[ax]))
            out_val = np.roll(out_val, indexes, ax)
        return Signal2D(out_val, index=self.index, columns=self.columns)

    def max_point(self):
        """
        Gets the (x, y) coordinates of the points that has the maximum amplitude.

        Returns
        -------
        x, y : ((2,) tuple)
            The (x, y) coordinates of the Signal2D maximum amplitude.
        """
        x = self.max(0).idxmax()
        y = self.loc[:, x].idxmax()
        return x, y

    def flatten(self):
        """
        Flattens the Signal2D to give coordinates (X, Y, Values).

        Returns
        -------
        x, y, z : numpy.ndarray
            The X-coordinates, Y-coordinates, Values of the Signal2D.
        """
        xv, yv = np.meshgrid(self.columns, self.index, indexing='xy')
        return np.array([xv.ravel(), yv.ravel(), self.values.ravel()])

    def remove_mean(self, axes=None):
        """
        Removes the mean along a given axis.

        Parameters
        ----------
        axes : string/int, optional
            The axis along  which to remove the means. If axis not specified, remove the global
            mean along all axes.

        Returns
        -------
        : Signal2D
            A copy of signal2D with means subtracted along given axes.
        """
        axes = self._get_axes_numbers(axes)
        out = self
        if 0 in axes:
            out = self - self.mean(0)
        if 1 in axes:
            out = (self.T - self.mean(1)).T
        return out

    def extract(self, axis=1, option='max'):
        """
        Extracts a 1-D Signal depending on the given option.

        Parameters
        ----------
        axis : int, optional
            Axis along which to extract the :class:`Signal`. Options are 0/'y'/'index' or
            1/'x'/'columns'.

        option : {'max'}, optional
            Currently only the option ``max`` is supported. This returns the signal at the
            maximum point in the Signal2D.

        Returns
        -------
        : Signal
            A new :class:`Signal` object representing the extracted signal.

        """
        axis = self._make_axes_as_num(axis)
        if len(axis) > 1:
            raise ValueError('axis cannot be None, or an array.')
        if option.lower() == 'max':
            x, y = self.max_point()
            if 0 in axis:
                out = self.loc[y, :]
                coord = y
            elif 1 in axis:
                out = self.loc[:, x]
                coord = x
            else:
                raise ValueError('Unknown axis value.')
        else:
            raise ValueError('Unknown option value.')
        return coord, out

    def tof(self, method='corr'):
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
        tof = []
        n = self.shape[1]
        if n < 2:
            ValueError('We need at least two echoes to compute the time of flight.')
        if method.lower() == 'corr':
            for i in np.arange(1, n):
                c = fftconvolve(self.iloc[:, i-1].values, self.iloc[:, i].values[::-1], mode='full')
                tof.append(self[i].ts * (self[i].size - np.argmax(c)))
            return np.array(tof)
        elif method.lower() == 'max':
            for i in self:
                tof.append(self[i].abs().idxmax())
            return np.diff(tof)
        elif method.lower() == 'thresh':
            for i in self:
                tof.append(self.limits(self[i])[0])
            return np.diff(tof)
        else:
            raise ValueError('method not supported. See documentation for supported methods.')

    def attenuation(self, a, d, c=None, f=None):
        """
        Compute the ultrasound attenuation. This is based on a piston transducer model. We assume
        that each column represents a different echo in the signal, and we compute attenuation
        across the columns.

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
        if self.shape[1] < 2:
            ValueError('We need at least two echoes to compute the attenuation.')

        # compute the propagation distance for each echo
        x = 2*d*np.arange(1, self.shape[1]+1)

        # estimate the mean velocity
        if c is None:
            c = np.mean(2*d/self.tof())

        # compute the FFT, which will be used for amplitudes computation
        Y = self.fft(ssb=True, axes=0).abs()

        # find the freuqnecies of the signal within the -6 dB bandwidth
        if f is None:
            # find the -6 dB limits of the frequency spectrum
            fmin, fmax = Y.iloc[:, 0].limits(0.5)
            amps = Y[fmin:fmax]
        elif f == 'max':
            fmin, fmax = Y.iloc[:, 0].limits(0.5)
            f = (fmin+fmax)/2
            amps = Y[f:f+self.ts[0]]
        else:
            if not hasattr(f, '__len__'):
                amps = Y[f:f+self.ts[0]]
                f = [f]
            else:
                amps = Y.loc[f, :]

        def compute_att(xloc, alpha, A, flocal):
            p = 2*np.pi*flocal*a**2/(c * xloc)
            D = 1 - np.exp(1j*p)*(jv(0, p) - 1j*jv(1, p))
            return A * abs(D) * np.exp(-alpha * xloc)

        params = pd.DataFrame(0, index=amps.index, columns=['alpha', 'A'])
        # params = pd.DataFrame(0, index=amps.index, columns=['alpha'])
        for fi in amps.index:
            val = amps.loc[fi, :].values
            # popt, pcov = curve_fit(lambda xp, alphap, ap: compute_att(xp, alphap, ap, fi), x,
            #                        val, p0=[0.02, 10])
            # params.loc[fi] = popt
            p1 = 2*np.pi*fi*a**2/(c * d)
            D1 = abs(1 - np.exp(1j*p1)*(jv(0, p1) - 1j*jv(1, p1)))
            p2 = 2*np.pi*fi*a**2/(c * 2*d)
            D2 = abs(1 - np.exp(1j*p2)*(jv(0, p2) - 1j*jv(1, p2)))
            params.loc[fi, 'alpha'] = (1/(2*d))*(np.log(val[0]/val[1]) - np.log(D1/D2))

        return params

    @property
    def ts(self):
        """ Get the signal sampling period. """
        idx = np.mean(np.diff(self.index)) if len(self.index) > 1 else 0
        col = np.mean(np.diff(self.columns)) if len(self.columns) > 1 else 0
        return idx, col

    @property
    def x(self):
        """ Convenience property to return X-axis coordinates as ndarray. """
        return self.axes[1]

    @property
    def y(self):
        """ Convenience property to return Y-axis coordinates as ndarray. """
        return self.axes[0]

    @property
    def extent(self):
        """ Returns the extents of the axes values"""
        return self.index.max() - self.index.min(), self.columns.max() - self.columns.min()

    @property
    def real(self):
        return Signal2D(self.real, index=self.index, columns=self.columns)

    @property
    def imag(self):
        return Signal2D(self.imag, index=self.index, columns=self.columns)


Signal2D._setup_axes(['index', 'columns'], info_axis=1, stat_axis=0,
                     axes_are_reversed=True, aliases={'rows': 0, 'y': 0, 'x': 1})
