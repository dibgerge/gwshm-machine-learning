""" Generates the complex features based on the center frequency """
import numpy as np
import pandas as pd
from packages import scihdf, utils, utkit


# read the configuration file
conf = utils.Configurations()['guided_waves']
store = scihdf.SciHDF(conf['data']['rx_dump'])

idx, out = [], []
for info in store:
    print(info)
    wave = utkit.Signal(store[info])
    nfft = 2**(int(np.log2(len(wave))) + 1)

    # remove the initial part of the wave (time of flight)
    wave = wave.loc[conf['features']['complex_amp']['holdoff']:]
    sxx = wave.spectrogram(width=conf['features']['complex_amp']['pw'],
                           overlap=conf['features']['complex_amp']['pw']/2,
                           nfft=nfft,
                           mode='complex')
    out.append(sxx.loc[conf['features']['complex_amp']['frequency']*1e3:].iloc[0])
    idx.append([info.sensor, info.impact, info.index])

# make the dataframe and sort its index
mux = pd.MultiIndex.from_arrays(list(zip(*idx)), names=['sensor', 'impact', 'index'])
features = pd.DataFrame(out, index=mux)
features.sort_index(axis=0, inplace=True)

# write to excel sheet
writer = pd.ExcelWriter(conf['features']['complex_amp']['dump'])
features.astype(str).to_excel(writer)
writer.save()
writer.close()
store.close()
