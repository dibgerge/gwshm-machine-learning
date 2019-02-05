"""
Compute the short-time coherence function as proposed by Michaels.
"""
from packages import utkit, scihdf, utils
import pandas as pd
import numpy as np
from scipy.signal import fftconvolve


def compute_coherence(s1, s2, width, overlap):
    # align signal s1 with s2
    s1 = s1(s2.index)

    time_ = s1.index[0]
    tstep = width - overlap

    if tstep <= 0:
        raise ValueError('overlap cannot be larger than width.')
    idx, out = [], []
    while time_ + width <= s1.index[-1]:
        s1_wind = s1.window(index1=time_, index2=time_+width, win_fcn='hann')
        s2_wind = s2.window(index1=time_, index2=time_+width, win_fcn='hann')
        c = fftconvolve(s1_wind, s2_wind[::-1])/np.sqrt(np.sum(s1_wind**2) * np.sum(s2_wind**2))
        out.append(np.max(c))
        idx.append(time_ + width/2)
        time_ += tstep
    return pd.Series(out, index=idx)


conf = utils.Configurations()
featconf = conf['journal_2017']['features']['coherence']
store = scihdf.SciHDF(conf['guided_waves']['data']['rx_dump'])

# get the baseline signals
idx, baseline = [], []
for info in store.keys_with(frequency=featconf['frequency'], actuator=featconf['actuator'],
                            impact=0):
    baseline.append(utkit.Signal(store[info]).loc[featconf['holdoff']:])
    idx.append([info.sensor, info.index])
mux = pd.MultiIndex.from_arrays(list(zip(*idx)), names=['sensor', 'index'])
baseline = pd.DataFrame(baseline, index=mux)
baseline.sort_index(axis=0, inplace=True)


idx, out = [], []
for info in store.keys_with(frequency=featconf['frequency'], actuator=featconf['actuator']):
    wave = utkit.Signal(store[info])
    # remove the initial part of the wave (time of flight)
    wave = wave.loc[featconf['holdoff']:]

    for i, b in baseline.loc[pd.IndexSlice[info.sensor, :]].iterrows():
        print('Index: ', info.index, 'Baseline index: ', i)
        out.append(compute_coherence(wave, utkit.Signal(b),
                                     width=featconf['pw'],
                                     overlap=featconf['pw']/2))
        idx.append([info.sensor, info.impact, i, info.index])

# make the dataframe and sort its index
mux = pd.MultiIndex.from_arrays(list(zip(*idx)),
                                names=['sensor', 'impact', 'baseline_index', 'index'])
features = pd.DataFrame(out, index=mux)
features.sort_index(axis=0, inplace=True)

# write to excel sheet
writer = pd.ExcelWriter(featconf['dump'])
features.to_excel(writer)
writer.save()
writer.close()
store.close()
