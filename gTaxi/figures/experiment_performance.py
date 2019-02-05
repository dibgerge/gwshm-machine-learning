from matplotlib import pyplot
import pandas as pd
import numpy as np
from packages import utils
from os.path import join
import matplotlib as mp

labs = 'abcdefghij'
conf = utils.Configurations()
prob = pd.read_excel(join(conf['data_root'], 'complexamp_performance.xlsx') , index_col=(0, 1, 2),
                     squeeze=True)

pyplot.style.use('plot_styles.mplstyle')
mp.rc('figure.subplot', left=0.07, top=0.88, bottom=0.13, right=0.99, hspace=0.21, wspace=0.13)
fig, axarr = pyplot.subplots(2, 5, figsize=(7, 4.2), dpi=72)

for ax, (groupID, data) in zip(*(axarr.ravel(), prob.unstack(level=0).groupby(level='impact'))):
    data.index = data.index.droplevel('impact')
    data = data.fillna(method='bfill')

    data.plot(ax=ax, legend=False,
              xticks=[4, 8, 12, 16, 20, 24],
              yticks=[0, 0.2, 0.4, 0.6, 0.8, 1],
              ms=6, mec='k', mew=0.5,
              style=['--o', '-s', '--v', '-^', '--<', '->'],
              grid=True)
    ax.set_title('Impact %d' % groupID, y=0.96)
    ax.set_ylim([-0.07, 1.05])

    if groupID % 5 != 0:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel('Probability')

    if groupID < 5:
        ax.set_xlabel('(%s)' % labs[groupID], labelpad=-5)
        ax.set_xticklabels([])
    else:
        ax.set_xlabel('# baselines\n(%s)' % labs[groupID])

    if groupID == 0:
        ax.axvspan(4, 24, color='grey', alpha=0.2)
    # ax.set_xticks(np.unique(prob.index.get_level_values(1).values))
    # break

axarr[0, 0].set_title('Baseline')
pyplot.figlegend(ax.lines, np.unique(prob.index.get_level_values(0).values),
                 loc='upper center', ncol=6, labelspacing=0.)
pyplot.savefig('experiment_performance.svg')
#pyplot.show()
