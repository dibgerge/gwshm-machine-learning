import numpy as np
from scipy.signal import hilbert


def operate(signal, option='', axis=0):
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
    yout = signal
    if 'e' in option:
        # make hilbert transform faster by computing it at powers of 2
        n = signal.shape[axis]
        pwr2 = np.log2(n)
        n = 2 ** int(pwr2) if pwr2.is_integer() else 2 ** (int(pwr2) + 1)
        yout = np.abs(hilbert(yout.values, N=n, axis=axis))
        yout = yout[:signal.shape[0], :signal.shape[1]]
    if 'n' in option:
        yout = yout / np.abs(yout).max().max()
    if 'd' in option:
        yout = 20 * np.log10(np.abs(yout))
    return Signal2D(yout, index=self.index, columns=self.columns)