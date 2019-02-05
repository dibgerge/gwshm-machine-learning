"""
Plot the images of the laser scans
"""
from packages import utils, utkit, scihdf
import matplotlib.pyplot as plt
import matplotlib as mpl


plt.style.use('plot_styles.mplstyle')
mpl.rc('figure.subplot', left=0.04, top=0.94, bottom=0.07, right=0.99, hspace=0.1, wspace=0.05)

conf = utils.Configurations()['laser']
store = scihdf.SciHDF(conf['dump'])

my_cmap = mpl.cm.get_cmap('bone')
my_cmap.set_under('w')

xlims, ylims = [60, 140], [20, 80]

fig, axarr = plt.subplots(2, 5, figsize=(7, 2.6), dpi=72)

for i, (ax, info) in enumerate(zip(*(axarr.ravel(), store))):
    print(info)
    scan = utkit.Signal2D(store[info])
    scan = scan.loc[ylims[0]:ylims[1], xlims[0]:xlims[1]]
    scan = scan.operate('n')

    # aspect = np.diff(ylims) / np.diff(xlims)
    # plt.axes([0,0,1,1])
    ax.set_aspect('equal')
    if i % 5 != 0:
        ax.get_yaxis().set_visible(False)
    if i < 5:
        ax.get_xaxis().set_visible(False)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.imshow(scan.values, cmap=my_cmap, vmax=1, vmin=0,
              extent=[xlims[0], xlims[1], ylims[0], ylims[1]],
              interpolation='nearest')

    ax.set_title('Impact %s' % info.impact, y=0.96)


# change the baseline image title
axarr[0, 0].set_title('Baseline')

plt.savefig('optical_images.svg')
#plt.show()
store.close()
