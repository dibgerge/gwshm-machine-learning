""" A schematic of the classification approach, showing the classifier along with the defect
signal."""
from packages import utkit, scihdf, utils
from matplotlib import pyplot
from os.path import join
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

np.random.seed(10)
baselines = utils.make_range_signals(15, [0.9, 1.1], [-np.pi/5, np.pi/5])
damage = utils.make_random_signals(2, 0.02, np.pi/10, 0.9)[0] + 0.2*np.exp(1j*np.pi/5)

pipe = Pipeline([('aug', utils.FeatureAugment()),
                 ('norm', StandardScaler()),
                 ('clf', utils.MultiOCSVM(eta=0.8, kernel='rbf', nu=0.01, gamma=0.4))])
pipe.fit(baselines)

pyplot.style.use(join('..', 'plot_styles.mplstyle'))
utils.plot_svm_frontiers(pipe, damage, figsize=(2, 2))
ax = pyplot.gca()
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xlabel('Real', labelpad=-4)
ax.set_ylabel('Imaginary', labelpad=-4)
pyplot.savefig('demo_classification.svg')
# pyplot.show()
