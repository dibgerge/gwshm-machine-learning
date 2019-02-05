""" Save the time signal and the feature vectors."""
from packages import utkit, scihdf, utils
from matplotlib import pyplot
from os.path import join
import numpy as np


# pyplot.style.use('plot_styles.mplstyle')
conf = utils.Configurations()['guided_waves']
store = scihdf.SciHDF(conf['data']['rx_dump'])
infos = store.keys_with(sensor='5T', impact=0)

# -- save the windowed signal
s = utkit.Signal(store[infos[0]])
t = np.linspace(100e-6, 137e-6, 50)
s = s(t).window(win_fcn='hann')
np.savetxt('out.txt', s.values)

# -- save the ffts
x, y = [], []
for info in infos:
    s = utkit.Signal(store[info]).window(index1=160e-6, index2=197e-6, win_fcn='hann')
    S = s.fft(ssb=True)
    x.append(S.real(100e3))
    y.append(S.imag(100e3))
x = np.array(x)
y = np.array(y)
ind = x.argsort()
newx = []
for xi in x[ind]:
    newx.append('"' + str(xi))
np.savetxt('ffts.txt', np.c_[newx, y[ind]], delimiter='\t',  fmt=('%s', '%s'))

# -- data aggregation
x = np.array(x) - np.array(x).reshape(-1, 1)
y = np.array(y) - np.array(y).reshape(-1, 1)
ind = x.ravel().argsort()
newx = []
for xi in x.ravel()[ind]:
    newx.append('"' + str(xi))
np.savetxt('fft_aggregates.txt', np.c_[newx, y.ravel().astype('str')[ind]], delimiter='\t',
           fmt=('%s', '%s'))
store.close()
