""" Show how we segment the signals into different bins."""
from packages import utkit, scihdf, utils
from matplotlib import pyplot
from os.path import join
import numpy as np
from matplotlib.ticker import FormatStrFormatter


pyplot.style.use('plot_styles.mplstyle')
conf = utils.Configurations()['guided_waves']

store = scihdf.SciHDF(conf['data']['rx_dump'])
info = scihdf.Info(sensor='5T', index=0, impact=0)

pw = conf['features']['complex_amp']['pw']
holdoff = conf['features']['complex_amp']['holdoff']

wave = utkit.Signal(store[info]).remove_mean()
store.close()

# make the figure
fig, ax = pyplot.subplots(2, 1, figsize=(3.4, 4),  dpi=100)
ax[0].plot(wave.index*1e6, wave)

i = 0
clrs = ['#020202', '#808080']
xticks = []
while holdoff + pw <= wave.index[-1]:
    wave_windowed = wave.window(index1=holdoff, index2=holdoff+pw, win_fcn='hann')
    p1, = ax[1].plot(wave_windowed.index*1e6, wave_windowed - 0.25*i, alpha=0.5,
                        color=clrs[i % 2])
    # if i % 4 == 0:
    #     ax[0].axvspan(1e6*holdoff, 1e6*(holdoff+pw), color=p1.get_color(), alpha=0.15)
    #     ax[1].axvspan(1e6*holdoff, 1e6*(holdoff+pw), color=p1.get_color(), alpha=0.15)
    holdoff += pw/2
    if i % 4 == 0:
        xticks.append(holdoff*1e6)
    i += 1

ax[1].set_xlabel('Time [$\mu s$]')
[axi.set_ylabel('Volts') for axi in ax.reshape(-1)]
[axi.grid() for axi in ax.reshape(-1)]
[axi.set_xticks(xticks) for axi in ax.reshape(-1)]
[axi.yaxis.set_major_formatter(FormatStrFormatter('%.1f')) for axi in ax.reshape(-1)]

pyplot.tight_layout(0, h_pad=0.05, rect=[0, 0, 0.975, 1])
ax[0].set_xlim([65, 400])
ax[1].set_xlim([65, 400])

pyplot.savefig('time_binning_example.svg')
# pyplot.show()
