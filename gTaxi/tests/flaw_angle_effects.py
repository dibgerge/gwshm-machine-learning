"""
Test what would happen if the defect angle is changing
"""
import numpy as np
from matplotlib import pyplot
from packages import utils
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, PredefinedSplit, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal as mvn, uniform
from sklearn.decomposition import PCA
import itertools

if __name__ == '__main__':
    # Means and standard deviations of phase and amplitude
    phi_mean, amp_mean = 0, 1
    phi_sigma, amp_sigma = [np.pi/10, np.pi/5, np.pi/4, np.pi/3, np.pi/2], [0, 0.05, 0.1, 0.2]

    # Correlation between phase and amplitude
    rho = 0.

    # Total number of defects to test
    n = 1000

    # Amplitude of the defect signals
    defect_amp = 2
    nbaselines = 20
    eta, gamma = 0.8, 0.2

    angs = np.linspace(-np.pi, np.pi, 30)

    # Make the SVM
    pipe = Pipeline([('aug', utils.FeatureAugment()),
                     ('norm', StandardScaler()),
                     ('clf', utils.MultiOCSVM(eta=eta, kernel='rbf', nu=0.01, gamma=gamma))])

    fig, axarr = pyplot.subplots(4, 5)

    for ax, (amp, phase) in zip(*(axarr.ravel(), itertools.product(amp_sigma, phi_sigma))):
    # for amp, phase in itertools.product(amp_sigma, phi_sigma):
        print(amp, phase/np.pi)
        # Generate the training data
        p = 3
        baseline = utils.make_range_signals(nbaselines,
                                            amp_range=[amp_mean - p * amp, amp_mean + p * amp],
                                            phase_range=[phi_mean - p * phase, phi_mean + p * phase],
                                            corr=1)
        # Train the SVM
        pipe.fit(baseline, np.ones(nbaselines))

        # rs = RandomizedSearchCV(pipe, {'clf__gamma': uniform(loc=0.3, scale=0.275),
        #                                'clf__eta': uniform(loc=0.7, scale=0.2)},
        #                         cv=LeaveOneOut(), n_iter=100, refit=True, n_jobs=-1)

        det, fa = [], []
        for ang in angs:
            # generate defect data
            healthy = utils.make_random_signals(n, amp, phase, rho)
            defects = healthy + defect_amp * np.exp(1j*np.ones(n)*ang)

            res = pipe.predict(defects)
            det.append(1-np.mean(res))
            res = pipe.predict(healthy)
            fa.append(1-np.mean(res))
            # utils.plot_svm_frontiers(pipe, defects)
            # utils.plot_svm_frontiers(pipe, healthy)
            # pyplot.show()

        ax.plot(angs/np.pi, det, '-o')
        ax.plot(angs/np.pi, fa, '-s')
        ax.set_title('Amp: %4.2f, Phase: %4.2f$\pi$' % (amp, phase/np.pi))
    #
    # # utils.plot_svm_frontiers(pipe, defects)
    pyplot.show()
