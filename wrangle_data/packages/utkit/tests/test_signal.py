import unittest
from utkit import Signal
import numpy as np
import pandas.util.testing as pdt
import numpy.testing as npt


class TestSignal(unittest.TestCase):
    fc = 1e6
    fs = 100e6
    n = 1000

    t = np.arange(n)/fs
    y = np.sin(2*np.pi*fc*t)

    s = Signal(y, index=t)

    def test_constructor_samplingintervalindex(self):
        """
        Does index input using sampling interval results in the same signal as inputing the
        whole time vector.
        """
        s1 = Signal(self.y, index=1/self.fs)
        s2 = Signal(self.y, index=self.t)
        pdt.assert_series_equal(s1, s2)

    def test_constructor_scalarsignal(self):
        s1 = Signal(1, index=[10.0])
        s2 = Signal(1, index=10.0)
        npt.assert_allclose(s1.index.values, (10.0,))
        npt.assert_allclose(s2.index.values, (10.0,))

    def test_constructor_emptyindex(self):
        s = Signal(np.arange(4))
        s2 = Signal(np.arange(4), index=np.arange(4))
        pdt.assert_series_equal(s, s2)

    def test_contrusctor_monotonicindex(self):
        """
        An error is raised if index is not monotonic.
        """
        self.assertRaises(Exception, Signal, [0, 1, 2], index=[-1, 2, 0])

    def test_window_allindices(self):
        t = np.arange(6)*1e-6
        s = Signal(np.ones(6), index=t)
        swindowed = Signal([0., 0., 1., 1., 1., 0.], index=t)
        s = s.window(index1=2e-6, index2=4e-6, win_fcn='boxcar')
        pdt.assert_series_equal(s, swindowed)

    def test_window_onlyindex1(self):
        t = np.arange(6)*1e-6
        s = Signal(np.ones(6), index=t)
        swindowed = Signal([0., 0., 1., 1., 1., 1.], index=t)
        s = s.window(index1=2e-6, win_fcn='boxcar')
        pdt.assert_series_equal(s, swindowed)

    def test_window_onlyindex2(self):
        t = np.arange(6)*1e-6
        s = Signal(np.ones(6), index=t)
        swindowed = Signal([1., 1., 1., 1., 1., 0.], index=t)
        s = s.window(index2=4e-6, win_fcn='boxcar')
        pdt.assert_series_equal(s, swindowed)

    def test_window_ispositional(self):
        t = np.arange(6)*1e-6
        s = Signal(np.ones(6), index=t)
        swindowed = Signal([0., 0., 1., 1., 1., 0.], index=t)
        s = s.window(index1=2, index2=4, is_positional=True, win_fcn='boxcar')
        pdt.assert_series_equal(s, swindowed)

    def test_window_ispositional2(self):
        t = np.arange(6)*1e-6
        s = Signal(np.ones(6), index=t)
        self.assertRaises(ValueError, s.window, index1=2e-6, index2=4e-6, is_positional=True)

    def test_window_noindices(self):
        t = np.arange(6)*1e-6
        s = Signal(np.ones(6), index=t)
        pdt.assert_series_equal(s, s.window(win_fcn='boxcar'))

    def test_window_fftbins(self):
        t = np.arange(6)*1e-6
        s = Signal(np.ones(6), index=t)
        self.assertRaises(IndexError, s.window, index1=2e-6, index2=4e-6, fftbins=True)

    def test_window_fftbins2(self):
        t = np.arange(9)*1e-6 - 4e-6
        s = Signal(np.ones(9), index=t)
        s = s.window(index1=0.99e-6, index2=3.1e-6, fftbins=True, win_fcn='boxcar')
        swindowed = Signal([0., 1., 1., 1., 0., 1., 1., 1., 0.], index=t)
        pdt.assert_series_equal(s, swindowed)

    def test_normalize_energy(self):
        t = np.arange(6)*1e-6
        s = Signal(np.ones(6), index=t)
        npt.assert_allclose(s.normalize('energy')[0], np.sqrt(1/5e-6))

    def test_normalize_max(self):
        t = np.arange(6)*1e-6
        s = Signal(2*np.ones(6), index=t)
        npt.assert_allclose(s.normalize('max')[0], (1.0,))

    def test_normalize_raisesexception(self):
        self.assertRaises(Exception, Signal().normalize, 'foo')

    def test_pad_right(self):
        s = Signal([0.1, 0.2, 0.3], index=[1e-6, 2e-6, 3e-6])
        sp = Signal([0.1, 0.2, 0.3, 0.0, 0.0], index=[1e-6, 2e-6, 3e-6, 4e-6, 5e-6])
        pdt.assert_series_equal(s.pad(4e-6, fill=0.0, position='right'), sp)

    def test_pad_left(self):
        s = Signal([0.1, 0.2, 0.3], index=[1e-6, 2e-6, 3e-6])
        sp = Signal([0.0, 0.0, 0.1, 0.2, 0.3], index=[-1e-6, 0.0, 1e-6, 2e-6, 3e-6])
        pdt.assert_series_equal(s.pad(4e-6, fill=0.0, position='left'), sp)

    def test_pad_split(self):
        s = Signal([0.1, 0.2, 0.3], index=[1e-6, 2e-6, 3e-6])
        sp = Signal([0.0, 0.1, 0.2, 0.3, 0.0], index=[0.0, 1e-6, 2e-6, 3e-6, 4e-6])
        pdt.assert_series_equal(s.pad(4e-6, fill=0.0, position='split'), sp)

    def test_pad_split2(self):
        s = Signal([0.1, 0.2, 0.3], index=[1e-6, 2e-6, 3e-6])
        sp = Signal([0.0, 0.1, 0.2, 0.3, 0.0, 0.0], index=[0.0, 1e-6, 2e-6, 3e-6, 4e-6, 5e-6])
        pdt.assert_series_equal(s.pad(5e-6, fill=0.0, position='split'), sp)

    def test_pad_fill_as_edge(self):
        s = Signal([0.1, 0.2, 0.3], index=[1e-6, 2e-6, 3e-6])
        sp = Signal([0.1, 0.1, 0.2, 0.3, 0.3, 0.3], index=[0.0, 1e-6, 2e-6, 3e-6, 4e-6, 5e-6])
        pdt.assert_series_equal(s.pad(5e-6, fill='edge', position='split'), sp)

    def test_segment(self):
        pass



if __name__ == '__main__':
    unittest.main()
