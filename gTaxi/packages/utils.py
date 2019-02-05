from os.path import join, abspath, dirname
import numpy as np
from . import utkit, scihdf
import yaml
from scipy.stats import multivariate_normal as mvn
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import binarize, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import six
from sklearn.pipeline import Pipeline
from matplotlib import pyplot
from abc import ABCMeta


def compute_pw(info, thresh=0.1):
    """
    Computes the pulse width of the excitation signal.
    
    Parameters
    ----------
    info : Info
        The info object for which to compute the corresponding excitation pulse width.

    thresh : float
        A number between (0, 1] selecting the threshold for which to compute the pulse width.

    Returns
    -------
    : float
        The pulse width in seconds.
    """
    conf = Configurations()

    # Open the excitation signals store
    excite_store = scihdf.SciHDF(conf['guided_waves']['data']['tx_dump'])
    excite = utkit.Signal(excite_store[info])
    pw = np.diff(excite.operate('e').limits(thresh=thresh))[0]
    excite_store.close()
    return pw


class Configurations:
    """
    Reads the project configurations files.
    """
    # the root directory where the code resides.
    code_dir = join(abspath(dirname(__file__)), '..')
    # project_dir = join(code_dir, '..', '..')

    # configuration file name
    fname = join(code_dir, 'project_config.yaml')

    def __init__(self):
        yaml.add_constructor('!join', self._join)
        with open(self.fname) as f:
            self._data = yaml.load(f)

    def __getitem__(self, item):
        return self._data[item]

    def _join(self, loader, node):
        """ Implements the join function for YAML files."""
        seq = loader.construct_sequence(node)
        return join(self.code_dir, *seq)


class FeatureAugment((six.with_metaclass(ABCMeta, BaseEstimator, TransformerMixin))):
    """
    This is a one to many transformation class. Given a set of baseline data, a single
    measurement is transformed to many by performing subtraction from each of the available
    baselines.
    """
    def __init__(self):
        self.baselines_ = None

    def fit(self, X, y=None):
        if X.dtype != np.complex:
            raise TypeError('X should be an array of complex numbers.')
        self.baselines_ = X
        return self

    def transform(self, X, y=None):
        if X.dtype != np.complex:
            raise TypeError('X should be an array of complex numbers.')
        scatter = X.reshape(-1, 1) - self.baselines_
        scatter = scatter.ravel()
        return np.c_[np.real(scatter), np.imag(scatter)]

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


class MultiOCSVM(OneClassSVM):
    """
    Extends One class SVM implementation of skLearn to support voting after the prediction stage.
    This is requires since this class assumes it takes data that has been transformed by the one
    to many transformer. .
    """
    def __init__(self, eta=1, kernel='rbf', degree=3, gamma='auto', coef0=0.0,
                 tol=0.001, nu=0.5, shrinking=True, cache_size=200, verbose=False, max_iter=-1,
                 random_state=None):
        """
        eta: float, (0., 1.]
            The minimum fraction of the one to many transformed samples before being able to
            classify as "healthy (or +1).

        Note:
        ----
        All other parameters are given to the :class:`OneClassSVM` in sklearn.
        """
        self.eta = eta
        self.mu_ = None
        self.sigma_ = None
        self.nbaseline = None
        self.cloud_ = None
        super().__init__(kernel, degree, gamma, coef0, tol, nu, shrinking, cache_size, verbose,
                         max_iter, random_state)

    def fit(self, X, y=None, **params):
        """
        Extends the :meth:`fit` for :class:`OneClassSVM`.
        """
        self.cloud_ = X
        self.nbaseline = int(np.sqrt(X.shape[0]))
        return super().fit(self.cloud_, y)

    def predict(self, X):
        """
        Unlike the underlying :class:`OneClassSVM`, this method returns decision as {0, 1},
        where 0 indicates "damage", and 1 indicates "healthy".
        """
        results = super().predict(X).reshape(-1, self.nbaseline)
        # make final decision by voting
        result = np.mean(binarize(results), axis=1) >= self.eta
        return result.astype(int)

    def score(self, X, y, **params):
        """
        This supports only two-class classification.
        """
        res = self.predict(X)
        return 1 - np.mean(abs(res-y))


def get_cov(amp_sigma=None, phase_sigma=None, corr=None):
    """
    Computes the covariance matrix between amplitude and phase, given their individual standard
    deviations and correlation.

    Parameters
    ----------
    amp_sigma : float
        A positive number indicating the amplitude standard deviation.

    phase_sigma : float
        A positive number indicating the phase standard deviation.

    corr : float, [0, 1]
        A number between 0 and 1 giving the correlation coefficient between phase and angle. A
        value of 0 means they are independent.

    Returns
    -------
    : array_like
        A 2x2 numpy covariance matrix.
    """
    if any(x is None for x in [amp_sigma, phase_sigma, corr]):
        return None
    cov = amp_sigma * phase_sigma * corr
    return np.array([[amp_sigma**2, cov], [cov, phase_sigma**2]])


def make_random_signals(n, amp_sigma, phase_sigma, corr, amp_mean=1, phase_mean=0):
    """
    Return baseline signals based on a multivariate random variable where the amplitude is
    correlated with the phase.

    Parameters:
    -----------
    n : int
        The number of signals to make.

    amp_sigma : float
        Amplitude standard deviation.

    phase_sigma : float
        Phase standard deviation

    corr : float, [0, 1]
        The correlation coefficient between amplitude and phase.

    amp_mean : float, optional
        The mean value for amplitude.

    phase_mean : float, optional
        The mean value for phase.

    Returns
    -------
    : array_like
        An array of size n of complex values.
    """
    cov = amp_sigma*phase_sigma*corr
    sigma = [[amp_sigma**2, cov], [cov, phase_sigma**2]]
    vals = mvn.rvs(mean=[amp_mean, phase_mean], cov=sigma, size=n)
    return vals[:, 0] * np.exp(1j * vals[:, 1])


def make_range_signals(n, amp_range, phase_range, corr=1):
    """
    Generates signals within a given range of amplitudes and phases.

    Parameters
    ----------
    n : int
        The number of signals to generate.

    amp_range : array_like
        An array of size 2 representing the start and end of amplitude range.

    phase_range : array_like
        An array of size 2 representing the start and end of phase range.

    corr : int, {-1, 1}
        If -1, then when amplitude increases phase decreases. If 1, both amplitude and phase
        increase.

    Returns
    -------
    : array_like
        An array of n complex values.
    """
    if corr not in [1, -1]:
        raise ValueError('corr should either be 1 or -1.')
    amps = np.linspace(amp_range[0], amp_range[1], n)
    phases = np.linspace(phase_range[0], phase_range[1], n)
    if corr == -1:
        phases = phases[::-1]
    return amps * np.exp(1j*phases)


def plot_svm_frontiers(pipe, defect_data=None, **kwargs):
    """
    Plots the contour and learned frontier for the :class:`MultiOCSVM` learning machine.

    Parameters
    ----------
    pipe : sklearn.pipeline.Pipeline
        The data transform and classification pipeline. It should contain three stages {'aug',
        'norm', 'clf'} corresponding to feature augmentation, data normalization,
        and classification.

    defect_data : array_like
        Any defect data to be plotted on top of the contours.
    """
    clf = pipe.named_steps['clf']
    xx, yy = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig = pyplot.figure(**kwargs)
    pyplot.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=pyplot.cm.Purples, alpha=0.5)
    pyplot.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')
    pyplot.plot(clf.cloud_[:, 0], clf.cloud_[:, 1], 'o', color='#9467bd',
                ms=3, mec='k', mew=0.25, alpha=0.85)
    pyplot.contour(xx, yy, Z, levels=[0], linewidths=1.75, colors='darkred', zorder=100)

    if defect_data is not None:
        trans = pipe.named_steps['aug'].transform(defect_data)
        trans = pipe.named_steps['norm'].transform(trans)
        pyplot.plot(trans[:, 0], trans[:, 1], 's', color='#2ca02c', mec='k',
                    ms=4, mew=0.25, alpha=0.8)
    return fig


def analytic_learner(nbaseline, amp_sigma, phase_sigma, amp_mean=1, phase_mean=0, eta=0.75,
                     nu=0.01, gamma=0.4, source='deterministic', n_std=1, corr=0):
    """
    This is a helper function to both build the training data (baseline signals), and fit the
    classifier pipeline.

    Parameters
    ----------
    nbaseline : int
        Number of baselines (training data) to generate.

    amp_sigma : float
        The standard deviation in the channel amplitude signal.

    phase_sigma : float
        The standard deviation in the channel phase signal.

    amp_mean : float, optional
        The mean amplitude of the channel.

    phase_mean : float, optional
        The mean phase of the channel.

    source : str, optional
        Determines how the baseline data is generated. If "deterministic", then the baseline
        amplitude/base span the space around the given mean values by some multiple of sigma. If
        "stochastic", the baselines are generated from an i.i.d. process where the covariance
        matrix represents the variance in the amplitude/phase.

    n_std : float, optional
        Only required if `source=deterministic`. Determines how many multiples of the channel
        variances do the baselines span.

    corr : float, optional
        Only required if `source=stochastic`. A value between [0, 1] representing the correlation
        between amplitude and phase in the i.i.d multi-variate channel generator.

    Returns
    -------
    pipe : sklearn.pipeline.Pipeline
        The fitted pipeline.

    Notes
    -----
    `eta`, `nu`, `gamma` are inputs to the :class:`MultiOCSVM`. See its docs for information
    about those parameters.
    """
    pipe = Pipeline([('aug', FeatureAugment()),
                     ('norm', StandardScaler()),
                     ('clf', MultiOCSVM(eta=eta, kernel='rbf', nu=nu, gamma=gamma))])

    if source == 'deterministic':
        baseline = make_range_signals(nbaseline,
                                      amp_range=[amp_mean-n_std*amp_sigma,
                                                 amp_mean+n_std*amp_sigma],
                                      phase_range=[phase_mean-n_std*phase_sigma,
                                                   phase_mean+n_std*phase_sigma], corr=1)
    elif source == 'stochastic':
        baseline = make_random_signals(nbaseline, amp_sigma, phase_sigma, corr,
                                       amp_mean=amp_mean, phase_mean=phase_mean)
    else:
        raise ValueError('source should be in {deterministic, stochastic}.')
    pipe.fit(baseline)
    return pipe


def analytic_classifier(pipe, n, amp_sigma, phase_sigma, defect_amp, defect_angle=0, amp_mean=1,
                        phase_mean=0, corr=0):
    """
    A helper function to generate data and classify them.

    Parameters
    ----------
    pipe : sklearn.pipeline.Pipeline
        The classification pipeline.
    n : int
        Number of samples to generate for classification.

    amp_sigma : float
    phase_sigma : float
    defect_amp : float
    defect_angle : float, optional
    amp_mean : float, optional
    phase_mean : float, optional
    corr : float, optional
        A value between [0, 1] representing the correlation between amplitude and phase in the i.i.d
         multi-variate channel generator.
    Returns
    -------
    pf, pd : float, float
        The computed probability of false alarm and probability of detection.
    """
    # generate defect data
    healthy = make_random_signals(n, amp_sigma, phase_sigma, corr, amp_mean, phase_mean)
    defects = healthy + defect_amp * np.exp(1j*np.ones(n)*defect_angle)

    defect_res = pipe.predict(defects)
    healthy_res = pipe.predict(healthy)
    return 1 - np.mean(healthy_res), 1 - np.mean(defect_res)


def gp(x, amp, phase, rho=0,  mean=None, gamma=10., kernel='gaussian'):
    if kernel == 'gaussian':
        C1 = amp**2 * np.exp(-gamma*(x[:, np.newaxis] - x[:, np.newaxis].T)**2)
        C2 = phase**2 * np.exp(-gamma*(x[:, np.newaxis] - x[:, np.newaxis].T)**2)
    elif kernel == 'laplacian':
        C1 = amp**2 * np.exp(-gamma*np.sqrt((x[:, np.newaxis] - x[:, np.newaxis].T)**2))
        C2 = phase**2 * np.exp(-gamma*np.sqrt((x[:, np.newaxis] - x[:, np.newaxis].T)**2))
    elif kernel == 'brownian':
        C1 = amp**2 * np.minimum(x[:, np.newaxis], x[:, np.newaxis].T)
        C2 = phase**2 * np.minimum(x[:, np.newaxis], x[:, np.newaxis].T)
    else:
        raise ValueError('Unknown kernel.')

    # C12 = rho * np.sqrt(np.linalg.det(C1)*np.linalg.det(C2))*np.ones_like(C1)
    C12 = rho * np.sqrt(C1*C2)
    cov = np.vstack((np.hstack((C1, C12)), np.hstack((C12, C2))))
    if mean is None:
        mu = np.concatenate((np.ones(len(x)), np.zeros(len(x))))
    else:
        mu = np.concatenate((np.ones(len(x))*mean[0], np.ones(len(x))*mean[1]))
    z = np.random.multivariate_normal(mu, cov, 1)[0]
    return z[:len(x)]*np.exp(1j*z[len(x):])


def gp_simple(x, sigma=10, mean=10, gamma=10., kernel='gaussian'):
    if kernel == 'gaussian':
        C = sigma * np.exp(-gamma*(x[:, np.newaxis] - x[:, np.newaxis].T)**2)
    elif kernel == 'laplacian':
        C = sigma * np.exp(-gamma * np.sqrt((x[:, np.newaxis] - x[:, np.newaxis].T) ** 2))
    elif kernel == 'brownian':
        C = sigma * np.minimum(x[:, np.newaxis], x[:, np.newaxis].T)
    else:
        raise ValueError('Unknown kernel.')
    z = np.random.multivariate_normal(mean * np.ones(len(x)), C, 1)
    return z[0]
