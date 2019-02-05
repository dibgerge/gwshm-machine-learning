import numpy as np
from matplotlib import pyplot
from packages import utils
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, PredefinedSplit, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal as mvn, uniform
import itertools
import matplotlib as mpl

if __name__ == '__main__':
    # Means and standard deviations of phase and amplitude
    phi_mean, amp_mean = 0, 1
    phi_sigma = np.linspace(np.pi/20, np.pi, 5)
    amp_sigma = np.linspace(0, 0.5, 5)

    # Correlation between phase and amplitude
    rho = 0.1

    # Total number of defects to test
    n = 1000

    # Amplitude of the defect signals
    defect_amp = 2
    nbaselines = 20
    eta, gamma = 0.8, 0.3

    # Make the SVM
    pipe = Pipeline([('aug', utils.FeatureAugment()),
                     ('norm', StandardScaler()),
                     ('clf', utils.MultiOCSVM(eta=eta, kernel='rbf', nu=0.01, gamma=gamma))])

    det = np.zeros((len(amp_sigma), len(phi_sigma)))
    fa = np.zeros_like(det)

    for i, amp in enumerate(amp_sigma):
        for j, phase in enumerate(phi_sigma):
            print(amp, phase/np.pi)
            # Generate the training data
            p = 4
            # baseline = utils.make_range_signals(nbaselines,
            #                                     amp_range=[amp_mean - p * amp, amp_mean + p * amp],
            #                                     phase_range=[phi_mean - p * phase, phi_mean + p * phase],
            #                                     corr=-1)
            baseline = utils.make_random_signals(nbaselines, amp, phase, rho)
            pipe.fit(baseline, np.ones(nbaselines))

            # generate defect data
            healthy = utils.make_random_signals(n, amp, phase, rho)
            defects = healthy + defect_amp * np.exp(1j*np.ones(n)*np.pi/2)

            res = pipe.predict(defects)
            det[i, j] = 1 - np.mean(res)

            res = pipe.predict(healthy)
            fa[i, j] = 1 - np.mean(res)
            # utils.plot_svm_frontiers(pipe)
            # pyplot.show()

    fig = pyplot.figure()
    p = pyplot.contourf(amp_sigma, phi_sigma, det.T)
    print(np.max(det))
    cax = fig.add_axes([0.9, 0.2, 0.02, 0.6])
    cb = mpl.colorbar.Colorbar(cax, p, cmap='viridis')

    fig = pyplot.figure()
    print(np.max(fa))
    p = pyplot.contourf(amp_sigma, phi_sigma, fa.T)
    cax = fig.add_axes([0.9, 0.2, 0.02, 0.6])
    cb = mpl.colorbar.Colorbar(cax, p, cmap='viridis')
    pyplot.show()
