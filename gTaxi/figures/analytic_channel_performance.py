import numpy as np
from matplotlib import pyplot
from packages import utils
import matplotlib as mp


def compute_performance(defect_amp, defect_angle, nsamples, channel_phase_sigma, channel_amp_sigma,
                        baseline_n_std, baseline_n=20, channel_rho=0,
                        eta=0.8, gamma=0.4, nu=0.01):
    """
    Generates analytical signals to assess performance under different channel conditions.

    Parameters
    ----------
    defect_amp : float
        Defect signal amplitude.

    defect_angle : float
        Defect signal phase.

    nsamples : int
        Number of samples generated for the defect signal. Used to generate monte-carlo samples
        fro assessing performance.

    channel_phase_sigma : array_like
        Channel phase variation conditions used to assess performance. The channel is assumed to
        be a multi-variate Gaussian. The variates of the channel are the baseline phase and
        amplitude.

    channel_amp_sigma : array_like
        Channel amplitude variation. Used in the computation of the channel covariance matrix.

    baseline_n_std : float
        Number of standard deviations for which the collected baselines cover the current
        channel conditions.

    baseline_n : int
        Number of baselines.

    channel_rho : float, [0, 1]
        The correlation coefficient between the channel phase and amplitude. Used to compute the
        channel covariance matrix.

    eta : float, [0, 1]
        Used for classification using one class SVM. The threshold of the classified signals
        which are classified as "healthy" before the final decision is given as "healthy".

    gamma : float
        The bandwidth for the SVM RBF kernel.

    nu : float
        Regularization coefficient for the SVM.

    Returns
    -------
    pf, pd : array_like, array_like
        The probability of false alarm and probability of detection. The sizes of the arrays is
        `len(channel_amp_sigma)` \times `len(channel_phase_sigma)`.
    """
    pd = np.zeros((len(channel_amp_sigma), len(channel_phase_sigma)))
    pf = np.zeros_like(pd)

    for i, amp in enumerate(channel_amp_sigma):
        for j, phase in enumerate(channel_phase_sigma):
            print('Amp: %4.2f, Phase: %4.2f' % (amp, phase/np.pi))

            pipe = utils.analytic_learner(baseline_n, amp, phase, eta=eta, nu=nu, gamma=gamma,
                                          source='deterministic', n_std=baseline_n_std)

            pf[i, j], pd[i, j] = utils.analytic_classifier(pipe, nsamples, amp, phase, defect_amp,
                                                           defect_angle, corr=channel_rho)
            print(pf[i, j], pd[i, j])
    return pf, pd


if __name__ == '__main__':
    amp_sigma = np.linspace(0, 0.6, 20)
    phase_sigma = np.linspace(np.pi / 25, np.pi, 20)

    # generate the 3 sigma data
    pf1, pd1 = compute_performance(defect_amp=2,
                                   nsamples=1000,
                                   defect_angle=np.pi,
                                   channel_phase_sigma=phase_sigma,
                                   channel_amp_sigma=amp_sigma,
                                   baseline_n_std=3,
                                   baseline_n=20,
                                   channel_rho=0.9,
                                   eta=0.8,
                                   gamma=0.05)

    # generate the 1 sigma data
    pf2, pd2 = compute_performance(defect_amp=2,
                                   nsamples=1000,
                                   defect_angle=np.pi,
                                   channel_phase_sigma=phase_sigma,
                                   channel_amp_sigma=amp_sigma,
                                   baseline_n_std=1,
                                   baseline_n=20,
                                   channel_rho=0.9,
                                   eta=0.8,
                                   gamma=0.05)

    # make the plot
    pyplot.style.use('plot_styles.mplstyle')
    mp.rc('figure.subplot', left=0.08, top=0.91, bottom=0.22, right=0.93, wspace=0.19)
    fig, ax = pyplot.subplots(1, 4, figsize=(7, 2.75), dpi=72)

    levels = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    titles = ['$P_D$, 3$\sigma$', '$P_{FA}$, 3$\sigma$', '$P_D$, 1$\sigma$', '$P_{FA}$, 1$\sigma$']
    for i, dat in enumerate([pd1, pf1, pd2, pf2]):
        p = ax[i].contourf(amp_sigma, phase_sigma/np.pi, dat.T + 1e-8,
                           levels=levels+1e-8, cmap=pyplot.cm.PuBu)
        ax[i].set_title(titles[i])

    [axi.set_yticklabels([]) for axi in ax[1:]]
    [axi.set_xticks([0, 0.2, 0.4, 0.6]) for axi in ax]
    [axi.set_xlabel('$\sigma_{amp}$\n %s' % lab) for lab, axi in zip(*(('(a)', '(b)', '(c)',
                                                                      '(d)'), ax))]
    ax[0].set_ylabel('$\sigma_{phase} \\times \pi$')

    cax = fig.add_axes([0.94, 0.25, 0.01, 0.65])
    pyplot.colorbar(p, cax=cax, cmap='viridis', ticks=levels+1e-8,
                    spacing='proportional')
    pyplot.savefig('analytical_channel_performance.svg')
    # pyplot.show()
