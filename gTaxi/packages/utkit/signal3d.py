import pandas as pd
import numpy as np
from scipy.signal import hilbert


class Signal3D(pd.Panel):
    """
    Represents data from raster scans. Each point is represented by its 2-D coordinates
    (X, Y), and contains a time series signal with time base t. By convention, the last axis is
    reserved for the time signals, and the first two axes compose the (X,Y) pairs.

    Since this is a specialized Ultrasound library, we make the following convention:
     * axis 0: The scan direction along the x-axis.
     * axis 1: The scan direction along the y-axis.
     * axis 2: The time series signals.

    """
    def __init__(self, data, items=None, major_axis=None, minor_axis=None, **kwargs):
        super().__init__(data=data, items=items, major_axis=major_axis, minor_axis=minor_axis,
                         **kwargs)
        self.items = self.items.astype(dtype=np.float64)
        self.major_axis = self.major_axis.astype(dtype=np.float64)
        self.minor_axis = self.minor_axis.astype(dtype=np.float64)

        for ax in self.axes:
            if not ax.is_monotonic_increasing:
                raise ValueError('Indices along all dimensions should be monotonically '
                                 'increasing. ')

    @property
    def _constructor(self):
        return Signal3D

    @property
    def _constructor_sliced(self):
        from .signal2d import Signal2D
        return Signal2D

    @classmethod
    def from_panel(cls, pnl):
        return cls(pnl.values, items=pnl.items, major_axis=pnl.major_axis,
                   minor_axis=pnl.minor_axis)

    def operate(self, option='', axis=2):
        """
        Operate on the signal along a given axis.

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
        : Signal3D
            The modified Signal3D.
        """
        axis = self._get_axis_number(axis)
        yout = self
        if 'e' in option:
            yout = np.abs(hilbert(yout, axis=axis))
        if 'n' in option:
            yout = yout/np.abs(yout).max().max()
        if 'd' in option:
            yout = 20*np.log10(np.abs(yout))
        return Signal3D(yout, items=self.items, major_axis=self.major_axis,
                        minor_axis=self.minor_axis)

    @staticmethod
    def _get_other_axes(axis):
        return np.setdiff1d([0, 1, 2], axis)

    def _get_axes_numbers(self, axes):
        if isinstance(axes, str):
            return self._get_axis_number(axes)
        elif hasattr(axes, '__len__'):
            return [self._get_axis_number(ax) for ax in axes]
        return axes

    @staticmethod
    def _cook_axes_args(val, axes):
        """
        Internal method to verify that a value is 3-element array. If it is a scalar or
        const:`None`, the missing values are selected based on the given *axis*.

        Parameters
        ---------
        val : float, array_like
            This is the value to be tested

        axes : int
            The axis for which the *val* corresponds.

        Returns
        -------
        : 2-tuple
            A 2-element array representing the filled *val* with missing axis value.
        """
        if axes is None:
            axes = np.arange(3)
        elif not hasattr(axes, '__len__'):
            axes = [axes]

        if not hasattr(val, '__len__'):
            val = np.ones(len(axes))*val

        if len(val) != len(axes):
            raise ValueError('value and axes have different lengths.')
        return val, axes

    def shift_axes(self, shift, axes=None, inplace=False):
        """
        Applies shifting of a given axis.

        Parameters
        ----------
        shift : float, array_like
            The value to shift the axis. If this is an array, it should be the same size as
            :attr:`axes`.

        axes : int, string, array_like, optional
            Specifies which of the axes to shift. To shift a single axis, :attr:`axes` is an
            int or string representing the dimension or the axis alias name. To shift
            multiple axes, :attr:`axes` can be an array of value or string names. If axes is
            :const:`None`, all three axes are shifted.

        inplace : bool
            Modify the current instance if True.

        Returns
        -------
        : Signal3D
            A copy of Signal3D with shifted axes, or the current instance if :attr:`inplace` is
            :const:`True`.
        """
        axes = self._get_axes_numbers(axes)
        shift, axes = self._cook_axes_args(shift, axes)
        out = Signal3D(self.values, items=self.items, major_axis=self.major_axis,
                       minor_axis=self.minor_axis) if not inplace else self
        for i, ax in enumerate(axes):
            out.set_axis(ax, out.axes[ax] - shift[i])
        return out

    def scale_axes(self, scale, axes=None, inplace=False):
        """
        Scale axes by a given amount.

        Parameters
        ----------
        scale : float, array_like
            The amount to scale the axes. If :attr:`scale` is an array, it should be the same
            size as :attr:`axes`.

        axes : int, string, array_like, optional
            Specifies which of the axes to shift. To shift a single axis, :attr:`axes` is an
            int or string representing the dimension or the axis alias name. To shift
            multiple axes, :attr:`axes` can be an array of value or string names. If axes is
            :const:`None`, all three axes are shifted.

        inplace : bool
            Modify the current instance if True.

        Returns
        -------
        : Signal3D
            A copy of Signal3D with scaled axes or the current instance if  :attr:`inplace` is
            :const:`True`.
        """
        axes = self._get_axes_numbers(axes)
        scale, axes = self._cook_axes_args(scale, axes)
        out = Signal3D(self.values, items=self.items, major_axis=self.major_axis,
                       minor_axis=self.minor_axis) if not inplace else self
        for i, ax in enumerate(axes):
            out.set_axis(ax, out.axes[ax]*scale[i])
        return out

    def skew(self, angle, skew_axis, other_axis):
        """
        Obtains the values for the axes after skewing by a certain angle, along a given plane.
        This method does not change the values of the object, only its axes.

        Parameters
        ----------
        angle : float
            The angle of the skew.

        skew_axis : int, string
            Name of the axis along which to skew the scan.

        other_axis: int, string
            Name of the second axis that forms the plane for skewing.

        Returns
        -------
        x, y : tuple
            A tuple representing the two axis value after skewing. Each element in the tuple is a
            matrix of shape of the current object along the plane formed by skew and other axis.
        """
        other_axis = self._get_axis_number(other_axis)
        skew_axis = self._get_axis_number(skew_axis)

        if other_axis == skew_axis:
            raise ValueError('other_axis cannot be the same as skew_axis.')

        slice_ax = self._get_other_axes([other_axis, skew_axis])[0]
        s = self.xs(self.axes[slice_ax][0], slice_ax)

        plane2d_axes = self._get_axes_numbers(self._get_plane_axes_index(slice_ax))
        x, y, _ = s.skew(angle, plane2d_axes.index(skew_axis), interpolate=False)
        return x, y

    def extract(self, option='max', axis=0):
        """
        Extracts a Signal2D according to a given option by slicing along a specified axis.

        Parameters
        ----------
        option : {'max', 'var', float} optional
            Select the method to find the slice
            * 'max': The maximum amplitude along the axis
            * 'var': the maximum variance along the axis.
            * scalar: A scalar can be specified to select a specific point along the axis

        axis : scalar/string, optional
            The axis along which to extract the slice.

        Returns
        -------
        : Signal2D
            A Signal2D object representing the slice.
        """
        axis = self._get_axis_number(axis)
        other_axes = self._get_other_axes(axis)

        if option == 'max':
            s = self.apply(lambda x: x.max().max(), axis=other_axes)
            option = s.idxmax()
        elif option == 'var':
            s = self.apply(lambda x: x.var().var(), axis=other_axes)
            option = s.idxmax()

        return option, self.xs(option, axis=axis)

    def dscan(self, option='max'):
        """
        Convenience method that return D-scans from Ultrasound Testing raster scans.
        Slices the raster scan along the y-t axes (i.e. at a given x location).

        Parameters
        ----------
        option : {'max', 'var', float}, optional
            Select the method to find the slice
            * 'max': The maximum amplitude along the axis
            * 'var': the maximum variance along the axis.
            * scalar: A scalar can be specified to select a specific point along the axis

        Returns
        -------
        : Signal2D
            Extracted D-scan as a Signal2D object.
        """
        return self.extract(option, axis=0)

    def bscan(self, option='max'):
        """
        Convenience method that return B-scans from Ultrasound Testing raster scans.
        Slices the raster scan along the x-t axes (i.e. at a given y location).

        Parameters
        ----------
        option : {'max', 'var', float} optional
            Select the method to find the slice
            * 'max': The maximum amplitude along the axis
            * 'var': the maximum variance along the axis.
            * scalar: A scalar can be specified to select a specific point along the axis

        Returns
        -------
        : Signal2D
            Extracted B-scan as a Signal2D object.
        """
        return self.extract(option, axis=1)

    def cscan(self, theta=None):
        """
        Specialized method for computing the C-Scans from Ultrasound Testing raster scans. This
        collapses the time axis and provides a top view of the scan, along the *x-y* axes.

        Parameters
        ----------
        theta : float
            The angle for which to skew the scan. This should be the wave propagation angle. The
            c-scan will be the top view after skew by the given angle.

        Returns
        -------
        : Signal2D
            The computed C-scan.
        """
        if theta is None:
            return self.max(axis=2)

        dx = self.ts[0]
        x, t = self.skew(theta, 'x', 'z')
        x /= dx
        x = x.astype(np.int32)
        xmin, xmax = np.min(x), np.max(x)
        nx = xmax - xmin + 1

        out = np.zeros((self.shape[1], nx))
        vals = np.abs(self.values)
        for i, coord in enumerate(range(xmin, xmax + 1)):
            ind0, ind1 = np.where(x == coord)
            out[:, i] = np.max(vals[ind1, :, ind0], axis=0)
        from . import Signal2D
        return Signal2D(out, index=self.major_axis, columns=np.arange(xmin, xmax+1)*dx)

    def flatten(self):
        """
        Flatten an array and its corresponding indices.

        Returns
        -------
        x, t, x, values : numpy.ndarray
            A 4-element tuple where each element is a flattened array of the Signal3D, and each
            representing a point with coordinates y, t, x and its value.
        """
        yv, tv, xv = np.meshgrid(self.Y, self.t, self.X, indexing='xy')
        return np.array([yv.ravel(), tv.ravel(), xv.ravel(), self.values.ravel()])

    def align(self, other, **kwargs):
        raise NotImplementedError

    def shift(self, periods=1, freq=None, axis='major'):
        raise NotImplementedError

    @property
    def ts(self):
        """
        The mean sampling interval for each of the axes.
        """
        return np.mean(np.diff(self.items)), np.mean(np.diff(self.major_axis)),\
               np.mean(np.diff(self.minor_axis))

    @property
    def x(self):
        """ Convenience property to return X-axis coordinates as ndarray. """
        return self.items.values

    @property
    def y(self):
        """ Convenience property to return Y-axis coordinates as ndarray. """
        return self.major_axis.values

    @property
    def z(self):
        """ Convenience property to return t-axis coordinates as ndarray. """
        return self.minor_axis.values

Signal3D._setup_axes(axes=['items', 'major_axis', 'minor_axis'], info_axis=0,
                     stat_axis=1, aliases={'major': 'major_axis',
                                           'minor': 'minor_axis',
                                           'x': 'items',
                                           'y': 'major_axis',
                                           'z': 'minor_axis'},
                     slicers={'major_axis': 'index',
                              'minor_axis': 'columns'})
