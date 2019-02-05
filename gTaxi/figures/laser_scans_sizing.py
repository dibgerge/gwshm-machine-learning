from packages import utils, utkit, scihdf
import numpy as np
from skimage.measure import regionprops, label
from skimage.morphology import closing, square
from matplotlib import pyplot
import matplotlib as mp
from os.path import join


conf = utils.Configurations()['laser']
store = scihdf.SciHDF(conf['dump'])
xlims, ylims = [60, 140], [20, 80]

damage_area = []
thresh = None
for info in store:
    scan = utkit.Signal2D(store[info])
    scan = scan.loc[ylims[0]:ylims[1], xlims[0]:xlims[1]]
    scan = scan.operate('n')

    if thresh is None:
        thresh = scan.min().min()*0.75

    bw = closing(scan.values < thresh, square(3))

    A = [region.area for region in regionprops(label(bw))]
    if not A:
        A = [0]
    damage_area.append(1e-2*np.max(A)*scan.ts[0]*scan.ts[1])

pyplot.style.use('plot_styles.mplstyle')
pyplot.figure(figsize=(3.3, 2.7))
pyplot.bar(np.arange(10), damage_area, alpha=0.9, width=0.5, edgecolor=(0, 0, 0, 1), linewidth=0.5)

for i, area in enumerate(damage_area):
    pyplot.text(i-0.33, area + .15, '%.1f' % area, color='k', fontsize=8)

ax = pyplot.gca()
ax.set_xticks(np.arange(10))
pyplot.grid(alpha=0.5)
pyplot.xlabel('Impact Number')
pyplot.ylabel('Delamination Diameter [$cm^2$]')
pyplot.ylim([0, 8])
pyplot.tight_layout(pad=0.2)
pyplot.savefig('laser_scans_sizing.svg')

#pyplot.show()
store.close()
