"""
Compute the false alarm rate and probability of detection of the experiments by using a different
number of baseline data.
"""
import numpy as np
import pandas as pd
from packages import scihdf, utils, utkit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot
from scipy.stats import multivariate_normal as mvn
from os.path import join

areas = []
cmap = pyplot.cm.viridis

# read the configuration file
conf = utils.Configurations()

feat = pd.read_excel(conf['guided_waves']['features']['complex_amp']['dump'], index_col=(0, 1, 2))
feat = feat.applymap(complex)

idx, out = [], []
for sid in conf['guided_waves']['experiment']['sensor_id']:
    for k in range(1, 7):
        baseline = feat.loc[pd.IndexSlice[sid, 0, :], :].values[::k, :]
        nbaseline, nbin = baseline.shape
        print(nbaseline)
        pipes = []
        for i in range(baseline.shape[1]):
            pipes.append(Pipeline([('aug', utils.FeatureAugment()),
                                   ('norm', StandardScaler()),
                                   ('clf', utils.MultiOCSVM(eta=0.55, kernel='rbf', nu=0.01,
                                                            gamma=0.05))]))
            pipes[-1].fit(baseline[:, i])

        for i in range(10):
            res = 0
            for j in range(nbin):
                defect = feat.loc[pd.IndexSlice[sid, i, :], :].values[:, j]
                res += pipes[j].predict(defect)

            res = (res >= 0.8*nbin).astype(int)
            out.append(1 - np.mean(res))
            idx.append([sid, nbaseline, i])

# make the dataframe and sort its index
mux = pd.MultiIndex.from_arrays(list(zip(*idx)), names=['sensor', 'nbaseline', 'impact'])
features = pd.Series(out, index=mux)
features.sort_index(axis=0, inplace=True)

# write to excel sheet
writer = pd.ExcelWriter(join(conf['data_root'], 'complexamp_performance.xlsx'))
features.to_excel(writer)
writer.save()
writer.close()
