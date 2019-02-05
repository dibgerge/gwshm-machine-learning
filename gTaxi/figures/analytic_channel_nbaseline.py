"""
Study the effects of the number of baselines using the analytic channel model.
"""
import numpy as np
from matplotlib import pyplot
from packages import utils
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib as mp


def compute_performance(defect_amp, defect_angle, nsamples, channel_phase_sigma, channel_amp_sigma,
                        baseline_n_std, baseline_n, channel_rho=0.0,
                        eta=0.8, gamma=0.4, nu=0.01):
    pd = np.zeros(len(baseline_n))
    pf = np.zeros_like(pd)

    for i, n in enumerate(baseline_n):
        print('Baselines: %d' % n)

        pipe = utils.analytic_learner(n, channel_amp_sigma, channel_phase_sigma,
                                      eta=eta, nu=nu, gamma=gamma,
                                      source='deterministic', n_std=baseline_n_std)

        pf[i], pd[i] = utils.analytic_classifier(pipe, nsamples,
                                                 channel_amp_sigma, channel_phase_sigma,
                                                 defect_amp, defect_angle, corr=channel_rho)
    return pf, pd


if __name__ == '__main__':
    nbaselines = np.arange(3, 30)
    pf1, pd1 = compute_performance(defect_amp=1.5,
                                   nsamples=5000,
                                   defect_angle=np.pi,
                                   channel_phase_sigma=np.pi/5,
                                   channel_amp_sigma=0.,
                                   baseline_n_std=3,
                                   baseline_n=nbaselines,
                                   channel_rho=0,
                                   eta=0.8,
                                   gamma=0.05)

    # make the plot
    # pyplot.style.use('plot_styles.mplstyle')
    # mp.rc('figure.subplot', left=0.08, top=0.91, bottom=0.15, right=0.93, wspace=0.19)
    # fig, ax = pyplot.subplots(1, 4, figsize=(7, 2.6))
    pyplot.plot(nbaselines, pf1, '-s')
    pyplot.plot(nbaselines, pd1, '-o')
    pyplot.show()

