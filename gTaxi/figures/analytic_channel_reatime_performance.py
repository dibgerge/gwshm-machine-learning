import numpy as np
from matplotlib import pyplot
from packages import utils
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from matplotlib import gridspec
import matplotlib as mp
from matplotlib.ticker import FormatStrFormatter


if __name__ == '__main__':
    # perform the rolling classification
    pipe = Pipeline([('aug', utils.FeatureAugment()),
                     ('norm', StandardScaler()),
                     ('clf', utils.MultiOCSVM(eta=0.2, kernel='rbf', nu=0.01, gamma=0.2))])

    np.random.seed(0)
    nbaselines = np.arange(3, 30)
    totfa, totpd = [], []

    for nbaseline in nbaselines:
        t = np.linspace(0, 1, nbaseline+1)
        pf, pd = [], []

        for j in range(1000):
            z = utils.gp(t, 0.1, np.pi/5, 0.5, kernel='brownian')
            pipe.fit(z[:-1])
            pf.append(pipe.predict(z[-1])[0])
            z[-1] = z[-1] + 1 * np.exp(1j*np.pi/4)
            pd.append(pipe.predict(z[-1])[0])

        totfa.append(1-np.mean(pf))
        totpd.append(1-np.mean(pd))

    pyplot.style.use('plot_styles.mplstyle')
    mp.rc('figure.subplot', left=0.14, top=0.98, bottom=0.15, right=0.97)
    pyplot.figure(figsize=(3.3, 2.75))

    pyplot.plot(nbaselines, totfa, '-o', ms=6, mec='k', label='False alarm')
    pyplot.plot(nbaselines, totpd, '-s', ms=6, mec='k', label='Detection')

    pyplot.ylabel('Probability')
    pyplot.xlabel('Window size [# baselines]')
    pyplot.legend()
    pyplot.grid()
    # pyplot.savefig('analytic_channel_realtime_performance.svg')
    pyplot.show()
