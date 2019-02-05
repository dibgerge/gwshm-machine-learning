from matplotlib import pyplot
import pandas as pd
import numpy as np
from packages import utils
from os.path import join
import matplotlib as mp

conf = utils.Configurations()
prob = pd.read_excel(join(conf['data_root'], 'experiment_rolling_performance.xlsx'),
                     index_col=(0, 1), squeeze=True)

pyplot.style.use('plot_styles.mplstyle')
mp.rc('figure.subplot', left=0.15, top=0.98, bottom=0.15, right=0.98, hspace=0.15, wspace=0.13)
fig = pyplot.figure(figsize=(3.3, 2.75), dpi=72)
ax = pyplot.gca()
prob.unstack(level=0).plot(ax=ax, kind='line',
                           ms=6, mec='k', mew=0.5,
                           xticks=np.arange(10),
                           grid=True,
                           style=['--o', '-s', '--v', '-^', '--<', '->'])
ax.set_ylabel('Probability')
#pyplot.show()
pyplot.savefig('experiment_rolling_win15.svg')
