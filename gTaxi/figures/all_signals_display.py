"""
Provide an example of the effects of temperature versus impact damage on the raw signals.
"""
from packages import utkit, scihdf, utils
from matplotlib import pyplot
import numpy as np
from os.path import join
import matplotlib as mp


def make_plot(ax, info_list, color):
    for info in info_list:
        s = utkit.Signal(store[info]).remove_mean()
        s.index *= 1e6
        ax.plot(s, '--', color=color, alpha=0.1, lw=0.5)
        ax.plot(s.operate('e'), '-', color=color, alpha=0.15)


impact = 5
# save lines for use in legend
baseline_clr, damage_clr = '#1f77b4', '#ff7f0e'

conf = utils.Configurations()['guided_waves']
store = scihdf.SciHDF(conf['data']['rx_dump'])

# set up figure
pyplot.style.use('plot_styles.mplstyle')
mp.rc('figure.subplot', left=0.07, top=0.84, bottom=0.16, right=0.98, wspace=0.1, hspace=0.2)
fig, axarr = pyplot.subplots(2, 3, figsize=(7, 3.4), dpi=72)
for i, (ax, sensor) in enumerate(zip(*(axarr.ravel(), conf['experiment']['sensor_id']))):
    print(sensor)
    info_baselines = store.keys_with(sensor=sensor, impact=0)
    info_impact = store.keys_with(sensor=sensor, impact=impact)

    make_plot(ax, info_baselines, color=baseline_clr)
    make_plot(ax, info_impact, color=damage_clr)

    ax.set_xlim([68, 400])
    ax.set_xticks([100, 200, 300, 400])
    ax.set_ylim([-1.2, 1.3])
    ax.set_title(sensor, y=0.97)
    if i < 3:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel('Time $[\mu s]$')

    if i % 3 != 0:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel('Volts')

    ax.grid()

# custom lines for use in legend
a, = pyplot.plot(np.NaN, np.NaN, color=baseline_clr)
b, = pyplot.plot(np.NaN, np.NaN, color=damage_clr)
pyplot.figlegend([a, b], ['Baseline', 'Impact %d' % impact],
                 loc='upper center', ncol=2, labelspacing=0.)

pyplot.savefig('all_signals_display.svg')
# pyplot.show()
store.close()
