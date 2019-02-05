import unittest
from .. import Signal3D
import numpy as np
import pandas.util.testing as pdt


class TestSignal3D(unittest.TestCase):
    _data = np.arange(27).reshape(3, 3, 3)
    s = Signal3D(_data)

    def test_shift_one_axis(self):
        """ :meth:`shift` test: shift one axis. """
        shifted = self.s.shift_axes(1, 0)
        s_true = Signal3D(self._data, items=np.arange(3)-1)
        pdt.assert_panel_equal(shifted, s_true)

    def test_shift_two_axes(self):
        """ :meth:`shift`: shift two axes with the same shift value."""
        shifted = self.s.shift_axes(1, (0, 1))
        s_true = Signal3D(self._data, items=np.arange(3)-1, major_axis=np.arange(3)-1)
        pdt.assert_panel_equal(shifted, s_true)

    def test_shift_three_axes(self):
        """ :meth:`shift`: shift three axes with the same shift value."""
        shifted = self.s.shift_axes(1)
        s_true = Signal3D(self._data, items=np.arange(3)-1, major_axis=np.arange(3)-1,
                          minor_axis=np.arange(3)-1)
        pdt.assert_panel_equal(shifted, s_true)

    def test_shift_three_axes2(self):
        """ :meth:`shift`: shift three axes with different shift values."""
        shift_vals = [1, 2, 3]
        shifted = self.s.shift_axes(shift_vals, (0, 1, 2))
        s_true = Signal3D(self._data, items=np.arange(3)-shift_vals[0],
                          major_axis=np.arange(3)-shift_vals[1],
                          minor_axis=np.arange(3)-shift_vals[2])
        pdt.assert_panel_equal(shifted, s_true)

    def test_scale_one_axis(self):
        """ :meth:`scale`: scale one axis."""
        scale = 2.0
        scaled = self.s.scale_axes(scale, axes=0)
        s_true = Signal3D(self._data, items=np.arange(3)*scale, major_axis=np.arange(3),
                          minor_axis=np.arange(3))
        pdt.assert_panel_equal(scaled, s_true)

    def test_scale_two_axes(self):
        """ :meth:`scale`: scale two axes."""
        scale = 2.0
        scaled = self.s.scale_axes(scale, axes=(0, 1))
        s_true = Signal3D(self._data, items=np.arange(3)*scale, major_axis=np.arange(3)*scale,
                          minor_axis=np.arange(3))
        pdt.assert_panel_equal(scaled, s_true)

    def test_scale_three_axes(self):
        """ :meth:`scale`: scale three axes."""
        scale = 2.0
        scaled = self.s.scale_axes(scale)
        s_true = Signal3D(self._data, items=np.arange(3)*scale, major_axis=np.arange(3)*scale,
                          minor_axis=np.arange(3)*scale)
        pdt.assert_panel_equal(scaled, s_true)

    def test_scale_three_axes2(self):
        """ :meth:`scale`: scale three axes."""
        scale = [1, 2, 3]
        scaled = self.s.scale_axes(scale, axes=(0, 1, 2))
        s_true = Signal3D(self._data, items=np.arange(3)*scale[0], major_axis=np.arange(3)*scale[1],
                          minor_axis=np.arange(3)*scale[2])
        pdt.assert_panel_equal(scaled, s_true)