import unittest
from .. import Signal, Signal2D
import numpy as np
import pandas.util.testing as pdt
import numpy.testing as npt


class TestSignal2D(unittest.TestCase):

    def test_constructor_scalarindex(self):
        s = Signal2D(np.arange(9).reshape(3, 3), index=2e-6, columns=[0., 1., 2.])
        s2 = Signal2D(np.arange(9).reshape(3, 3), index=[0, 2e-6, 4e-6], columns=[0., 1., 2.])
        pdt.assert_frame_equal(s, s2)

    def test_constructor_scalarindexfromdict(self):
        s = Signal2D({1: [0, 1, 3]}, index=2e-6)
        s2 = Signal2D([[0], [1], [3]], index=[0, 2e-6, 4e-6], columns=[1])
        pdt.assert_frame_equal(s, s2)

    def test_constructor_scalarcolumns(self):
        s = Signal2D(np.arange(9).reshape(3, 3), index=[0, 2e-6, 4e-6], columns=1.0)
        s2 = Signal2D(np.arange(9).reshape(3, 3), index=[0, 2e-6, 4e-6], columns=[0., 1., 2.])
        pdt.assert_frame_equal(s, s2)

    def test_constructor_scalarbothaxes(self):
        s = Signal2D(np.arange(9).reshape(3, 3), index=2e-6, columns=1.0)
        s2 = Signal2D(np.arange(9).reshape(3, 3), index=[0, 2e-6, 4e-6], columns=[0., 1., 2.])
        pdt.assert_frame_equal(s, s2)

    def test_constructor_scalarbothaxesFromdict(self):
        s = Signal2D({1.0: [0, 1, 3], 2.0: [4, 5, 6]}, index=2e-6)
        s2 = Signal2D([[0, 4], [1, 5], [3, 6]], index=[0, 2e-6, 4e-6], columns=[1.0, 2.0])
        pdt.assert_frame_equal(s, s2)

    def test_slicing(self):
        s = Signal2D(np.arange(9).reshape(3, 3), index=np.arange(3.), columns=np.arange(3.)*0.1)
        self.assertTrue(type(s[0.1]) == Signal)

    # def test_slicing2(self):
    #     s = Signal2D(np.arange(9).reshape(3, 3), index=np.arange(3.), columns=np.arange(3.)*0.1)
    #     self.assertTrue(type(s.loc[:, 0.1]) == Signal)
    #     self.assertTrue(type(s.loc[1., :]) == Signal)
    #
    # def test_slicing3(self):
    #     s = Signal2D(np.arange(9).reshape(3, 3), index=np.arange(3.), columns=np.arange(3.)*0.1)
    #     print(type(s.iloc[1, :]))
    #     self.assertTrue(type(s.iloc[:, 1]) == Signal)
    #     self.assertTrue(type(s.iloc[1, :]) == Signal)

    def test_shift_axis_axisnone(self):
        s = Signal2D(np.ones(12).reshape(4, 3), index=[0.0, 1.0, 2.0, 3.0], columns=[0., 1., 2.])
        s2 = Signal2D(np.ones(12).reshape(4, 3), index=[-1., 0., 1., 2.], columns=[-1., 0., 1.])
        pdt.assert_frame_equal(s.shift_axes(1), s2)

    def test_shift_axis_axisnone2(self):
        s = Signal2D(np.ones(12).reshape(4, 3), index=[0.0, 1.0, 2.0, 3.0], columns=[0., 1., 2.])
        s2 = Signal2D(np.ones(12).reshape(4, 3), index=[-1., 0., 1., 2.], columns=[-2., -1., 0.])
        pdt.assert_frame_equal(s.shift_axes([1, 2]), s2)

    def test_shift_axis_axisindex(self):
        s = Signal2D(np.ones(12).reshape(4, 3), index=[0.0, 1.0, 2.0, 3.0], columns=[0., 1., 2.])
        s2 = Signal2D(np.ones(12).reshape(4, 3), index=[-1.0, 0.0, 1.0, 2.0], columns=[0., 1., 2.])
        pdt.assert_frame_equal(s.shift_axes(1, axes='y'), s2)

    def test_shift_axis_axisindex2(self):
        s = Signal2D(np.ones(12).reshape(4, 3), index=[0.0, 1.0, 2.0, 3.0], columns=[0., 1., 2.])
        s2 = Signal2D(np.ones(12).reshape(4, 3), index=[-1., 0., 1., 2.], columns=[0., 1., 2.])
        pdt.assert_frame_equal(s.shift_axes(1, axes=0), s2)

    def test_shift_axis_columns(self):
        s = Signal2D(np.ones(12).reshape(4, 3), index=[0.0, 1.0, 2.0, 3.0], columns=[0., 1., 2.])
        s2 = Signal2D(np.ones(12).reshape(4, 3), index=[0.0, 1.0, 2.0, 3.0], columns=[-1., 0., 1.])
        pdt.assert_frame_equal(s.shift_axes(1, axes='x'), s2)

    def test_shift_axis_columns2(self):
        s = Signal2D(np.ones(12).reshape(4, 3), index=[0.0, 1.0, 2.0, 3.0], columns=[0., 1., 2.])
        s2 = Signal2D(np.ones(12).reshape(4, 3), index=[0.0, 1.0, 2.0, 3.0], columns=[-1., 0., 1.])
        pdt.assert_frame_equal(s.shift_axes(1, axes=1), s2)

    def test_scale_axis_axisnone_startstopnone(self):
        s = Signal2D(np.ones(12).reshape(4, 3), index=np.arange(4.), columns=np.arange(3.))
        s2 = Signal2D(np.ones(12).reshape(4, 3), index=[0., 0.5, 1., 1.5], columns=[0., 0.5, 1.0])
        pdt.assert_frame_equal(s.scale_axes(0.5), s2)

    def test_scale_axis_axisnone_startstopnone2(self):
        s = Signal2D(np.ones(12).reshape(4, 3), index=np.arange(4.), columns=np.arange(3.))
        s2 = Signal2D(np.ones(12).reshape(4, 3), index=[0., 0.5, 1., 1.5], columns=[0., 0.25, 0.5])
        pdt.assert_frame_equal(s.scale_axes([0.5, 0.25]), s2)

    def test_window_allindices(self):
        t = np.arange(6)*1e-6
        x = np.arange(6)*1e-6

        s = Signal2D(np.ones((6, 6)), index=t, columns=x)
        swindowed = Signal2D([[0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0.],
                              [0., 0., 1., 1., 1., 0.],
                              [0., 0., 1., 1., 1., 0.],
                              [0., 0., 1., 1., 1., 0.],
                              [0., 0., 0., 0., 0., 0.]], index=t, columns=x)

        s = s.window(index1=2e-6, index2=4e-6, win_fcn='boxcar')
        pdt.assert_frame_equal(s, swindowed)

    def test_window_allindices2(self):
        t = np.arange(6)*1e-6
        x = np.arange(6)*1e-3

        s = Signal2D(np.ones((6, 6)), index=t, columns=x)
        swindowed = Signal2D([[0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0.],
                              [0., 0., 1., 1., 1., 1.],
                              [0., 0., 1., 1., 1., 1.],
                              [0., 0., 1., 1., 1., 1.],
                              [0., 0., 0., 0., 0., 0.]], index=t, columns=x)

        s = s.window(index1=[2e-6, 2e-3], index2=[4e-6, 5e-3], win_fcn='boxcar')
        pdt.assert_frame_equal(s, swindowed)

    def test_window_axis1(self):
        t = np.arange(6)*1e-6
        x = np.arange(6)*1e-3

        s = Signal2D(np.ones((6, 6)), index=t, columns=x)
        swindowed = Signal2D([[0., 0., 1., 1., 1., 0.],
                              [0., 0., 1., 1., 1., 0.],
                              [0., 0., 1., 1., 1., 0.],
                              [0., 0., 1., 1., 1., 0.],
                              [0., 0., 1., 1., 1., 0.],
                              [0., 0., 1., 1., 1., 0.]], index=t, columns=x)

        s = s.window(index1=2e-3, index2=4e-3, axes=1, win_fcn='boxcar')
        pdt.assert_frame_equal(s, swindowed)

    def test_window_axis0(self):
        t = np.arange(6) * 1e-6
        x = np.arange(6) * 1e-3

        s = Signal2D(np.ones((6, 6)), index=t, columns=x)
        swindowed = Signal2D([[0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0.],
                              [1., 1., 1., 1., 1., 1.],
                              [1., 1., 1., 1., 1., 1.],
                              [1., 1., 1., 1., 1., 1.],
                              [0., 0., 0., 0., 0., 0.]], index=t, columns=x)

        s = s.window(index1=2e-6, index2=4e-6, axes=0, win_fcn='boxcar')
        pdt.assert_frame_equal(s, swindowed)

    def test_window_ispositional(self):
        t = np.arange(6) * 1e-6
        x = np.arange(6) * 1e-3

        s = Signal2D(np.ones((6, 6)), index=t, columns=x)
        swindowed = Signal2D([[0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0.],
                              [1., 1., 1., 1., 1., 1.],
                              [1., 1., 1., 1., 1., 1.],
                              [1., 1., 1., 1., 1., 1.],
                              [0., 0., 0., 0., 0., 0.]], index=t, columns=x)

        s = s.window(index1=2, index2=4, axes=0, is_positional=True, win_fcn='boxcar')
        pdt.assert_frame_equal(s, swindowed)

    def test_window_ispositional2(self):
        t = np.arange(6) * 1e-6
        x = np.arange(6) * 1e-3

        s = Signal2D(np.ones((6, 6)), index=t, columns=x)
        swindowed = Signal2D([[0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 1., 1., 1.],
                              [0., 0., 0., 1., 1., 1.],
                              [0., 0., 0., 1., 1., 1.],
                              [0., 0., 0., 0., 0., 0.]], index=t, columns=x)

        s = s.window(index1=[2, 3], index2=[4, 5], is_positional=True, win_fcn='boxcar')
        pdt.assert_frame_equal(s, swindowed)

    def test_window_fftbins(self):
        t = np.arange(7) * 1e-6 - 3e-6
        x = np.arange(6) * 1e-3
        s = Signal2D(np.ones((7, 6)), index=t, columns=x)

        swindowed = Signal2D([[0., 0., 0., 0., 0., 0.],
                              [1., 1., 1., 1., 1., 1.],
                              [1., 1., 1., 1., 1., 1.],
                              [0., 0., 0., 0., 0., 0.],
                              [1., 1., 1., 1., 1., 1.],
                              [1., 1., 1., 1., 1., 1.],
                              [0., 0., 0., 0., 0., 0.]], index=t, columns=x)

        s = s.window(index1=0.9e-6, index2=2.1e-6, axes=0, fftbins=True, win_fcn='boxcar')
        pdt.assert_frame_equal(s, swindowed)

    def test_window_fftbins2(self):
        t = np.arange(7) * 1e-6
        x = np.arange(6) * 1e-3
        s = Signal2D(np.ones((7, 6)), index=t, columns=x)
        self.assertRaises(IndexError, s.window, index1=0.9e-6, index2=2.1e-6, axes=0,
                          fftbins=True, win_fcn='boxcar')

    def test_pad_oneaxis(self):
        t = np.arange(4) * 1e-6
        x = np.arange(3) * 1e-3
        s = Signal2D(np.ones((4, 3)), index=t, columns=x)

        spadded = Signal2D([[1., 1., 1., 0., 0., 0., 0.],
                            [1., 1., 1., 0., 0., 0., 0.],
                            [1., 1., 1., 0., 0., 0., 0.],
                            [1., 1., 1., 0., 0., 0., 0.]], index=t, columns=np.arange(7)*1e-3)
        pdt.assert_frame_equal(s.pad(6e-3, axes=1, fill=0., position='end'), spadded)

    def test_pad_oneaxis2(self):
        t = np.arange(4) * 1e-6
        x = np.arange(3) * 1e-3
        s = Signal2D(np.ones((4, 3)), index=t, columns=x)

        spadded = Signal2D([[0., 0., 0., 0., 1., 1., 1.],
                            [0., 0., 0., 0., 1., 1., 1.],
                            [0., 0., 0., 0., 1., 1., 1.],
                            [0., 0., 0., 0., 1., 1., 1.]],
                           index=t, columns=np.arange(7)*1e-3-4e-3)
        pdt.assert_frame_equal(s.pad(6e-3, axes=1, fill=0., position='start'), spadded)

    def test_get_axes_numbers_two_axes(self):
        """:meth:`_get_axes_numbers` test"""
        s = Signal2D(np.arange(10))
        ax = s._get_axes_numbers(['x', 'y'])
        self.assertEqual(ax, [1, 0])

    def test_get_axes_numbers_two_axes2(self):
        """:meth:`_get_axes_numbers` test"""
        s = Signal2D(np.arange(10))
        ax = s._get_axes_numbers([1, 0])
        self.assertEqual(ax, [1, 0])

    def test_get_axes_numbers_none_axis(self):
        """:meth:`_get_axes_numbers` test"""
        s = Signal2D(np.arange(10))
        ax = s._get_axes_numbers(None)
        self.assertEqual(ax, [0, 1])

    def test_get_axes_numbers_one_axis(self):
        """:meth:`_get_axes_numbers` test"""
        s = Signal2D(np.arange(10))
        ax = s._get_axes_numbers('x')
        self.assertEqual(ax, [1])

    def test_get_axes_numbers_one_axis2(self):
        """:meth:`_get_axes_numbers` test"""
        s = Signal2D(np.arange(10))
        ax = s._get_axes_numbers(1)
        self.assertEqual(ax, [1])

    def test_fft(self):
        pass