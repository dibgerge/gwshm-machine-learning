"""
Shows an example of the guided waves time signals. This is useful just to see how the signal
looks like, and to find a start time for computing the spectrogram, etc...

Plots all six sensors, two impacts and all temperatures for each impact

"""
from packages import utils, utkit, scihdf
from matplotlib import pyplot

# impact level to plot, along with baseline
impact = 8
conf = utils.get_configuration()['guided_waves']
store = scihdf.SciHDF(conf['data']['rx_dump'])
cmaps = ['Blues', 'Reds']

fig, ax = pyplot.subplots(2, 3)

for i, sid in enumerate(conf['experiment']['sensor_id']):
    r, c = int(i/3), i % 3
    for j, im in enumerate([0, impact]):
        waves = utkit.Signal2D()
        for info in store.keys_with(actuator='2T', frequency=100, sensor=sid, impact=im):
            waves[info.index] = store[info]
        waves.sort_index(axis=1, inplace=True)
        waves.index *= 1e6
        waves.plot(ax=ax[r, c], legend=False, cmap=cmaps[j], alpha=0.4)
    ax[r, c].set_xlabel('Time [$\mu s$]')
    ax[r, c].set_ylabel('Volts')
    ax[r, c].grid(alpha=0.5)
    ax[r, c].set_title(sid)
pyplot.show()
store.close()