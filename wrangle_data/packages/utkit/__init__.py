"""
.. currentmodule:: utkit

.. sectionauthor:: Gerges Dib <dibgerge@gmail.com>
.. codeauthor:: Gerges Dib <dibgerge@gmail.com>

:mod:`utkit` extends the data structures in the popular library :mod:`pandas` in order to
support short time signals generally encountered in wave propagation measurements. The main
purpose of this library is to implement signal processing procedures for *signal shaping*,
*transforms*, and *feature extraction* in a simple manner. All the while, keeping track and managing
signal values and their corresponding indices (time/space values).

The library supports three data structures:

.. autosummary::
    :nosignatures:
    :toctree: generated/

    Signal
    Signal2D
    Signal3D
"""
from .signal import *
from .signal2d import *
from .signal3d import *
# from . import peakutils


__version__ = 0.2
