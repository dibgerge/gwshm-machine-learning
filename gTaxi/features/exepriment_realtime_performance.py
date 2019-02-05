from packages import utkit, utils, scihdf
import pandas as pd
import numpy as np
from matplotlib import pyplot
from os.path import join
import matplotlib as mp
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from concurrent import futures
import itertools

# setup the classifier hyperparameters
window_size = 15
zeta = 0.8
n_mc = 1000  # number of monte-carlo simulations

conf = utils.Configurations()
store = pd.read_excel(join(conf['data_root'], 'gw_features_complexamp.xlsx'),
                      index_col=(0, 1, 2), squeeze=True).applymap(complex)


def get_signals(sensor, impact):
    s = store.loc[pd.IndexSlice[sensor, impact, :], :]
    s.index = s.index.droplevel(['impact', 'sensor'])
    rpart = utkit.Signal2D(np.real(s), index=s.index, columns=s.columns)
    ipart = utkit.Signal2D(np.imag(s), index=s.index, columns=s.columns)
    return rpart, ipart


def main(args):
    sensor, impact = args
    print(sensor, impact)

    rbase, ibase = get_signals(sensor, 0)
    nbaseline, nbin = rbase.shape

    rsig, isig = get_signals(sensor, impact)
    t = np.linspace(0, 1, window_size + 1)

    # Define the classification pipeline
    pipe = Pipeline([('aug', utils.FeatureAugment()),
                     ('norm', StandardScaler()),
                     ('clf', utils.MultiOCSVM(eta=0.2, kernel='rbf', nu=0.01, gamma=0.2))])

    det = []
    # perform monte carlo
    for i in range(n_mc):
        temperatures = utils.gp_simple(t, sigma=150, mean=10, kernel='laplacian')
        temperatures[temperatures < 0] = 0
        temperatures[temperatures > nbaseline - 1] = nbaseline - 1

        res = 0
        for j in range(nbin):
            # training
            rpart = rbase.iloc[:, j](temperatures[:-1], ext=3).values
            ipart = ibase.iloc[:, j](temperatures[:-1], ext=3).values
            pipe.fit(rpart + 1j * ipart)

            # prediction
            rpart = rsig.iloc[:, j](temperatures[-1], ext=3)
            ipart = isig.iloc[:, j](temperatures[-1], ext=3)
            res += pipe.predict(np.array([rpart + 1j * ipart]))[0]
        det.append((res >= zeta * nbin).astype(int))
    return (sensor, impact), 1 - np.mean(det)


if __name__ == '__main__':
    sensor_ids = conf['guided_waves']['experiment']['sensor_id']
    impacts = range(10)
    idx, out = [], []
    with futures.ProcessPoolExecutor() as pool:
        for result in pool.map(main, itertools.product(sensor_ids, impacts)):
            idx.append(result[0])
            out.append(result[1])

    # make the dataframe and sort its index
    mux = pd.MultiIndex.from_arrays(list(zip(*idx)), names=['sensor', 'impact'])
    features = pd.Series(out, index=mux)
    features.sort_index(axis=0, inplace=True)

    # write to excel sheet
    writer = pd.ExcelWriter(join(conf['data_root'], 'experiment_rolling_performance.xlsx'))
    features.to_excel(writer)
    writer.save()
    writer.close()
