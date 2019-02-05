from packages import utils
import numpy as np
from matplotlib import pyplot

b = 1

A = 1
phi = np.linspace(-np.pi, np.pi/2, 40)

for i in np.linspace(-np.pi, np.pi, 5):
    r = A * np.exp(1j*phi) + 2*np.exp(1j*i)

# pyplot.arrow(0, 0, np.real(b), np.imag(b), width=0.05)
# for ri in r:
#     pyplot.arrow(0, 0, np.real(ri), np.imag(ri), width=0.025, color='r')
    pyplot.plot(np.real(r - b), np.imag(r - b), 'o')


pyplot.plot(0, 0, 's', ms=10)
pyplot.grid()
pyplot.axes().set_aspect('equal')
pyplot.xlim([-3, 3])
pyplot.ylim([-3, 3])


# A = np.linspace(0.5, 1.5, 10)
# phi = np.linspace(-np.pi, np.pi, 10)
# r = A * np.exp(1j*phi)
#
# pyplot.figure()
# # pyplot.arrow(0, 0, np.real(b), np.imag(b), width=0.05)
# # for ri in r:
# #     pyplot.arrow(0, 0, np.real(ri), np.imag(ri), width=0.025, color='r')
# pyplot.plot(np.real(r - b), np.imag(r - b), 'o')
# pyplot.xlim([-2, 2])
pyplot.show()