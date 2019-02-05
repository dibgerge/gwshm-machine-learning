import numpy as np
from matplotlib import pyplot
from packages import utils
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from matplotlib import gridspec
import matplotlib as mp
from matplotlib.ticker import FormatStrFormatter


if __name__ == '__main__':
    np.random.seed(0)
    nbaseline = 10
    tdamage = 50
    t = np.arange(0, 1, 0.01)
    z = utils.gp(t, 0.1, np.pi/5, 0.5, kernel='brownian')
    z[tdamage:] = z[tdamage:] + 0.5 * np.exp(1j*np.ones(len(z[tdamage:]))*np.pi/4)

    # initialize the figure
    pyplot.style.use('plot_styles.mplstyle')
    mp.rc('figure.subplot', left=0.2, top=0.98, bottom=0.15, right=0.85)
    fig = pyplot.figure(figsize=(3.5, 2.75))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3], hspace=0.1)

    # draw the amplitude
    ax1 = pyplot.subplot(gs[1])
    ax1.plot(t, np.abs(z), color='#1f77b4')
    ax1.set_xlabel('slow time')
    ax1.tick_params('y', colors='#1f77b4')
    ax1.set_ylabel('Amplitude', color='#1f77b4')
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax1.set_xlim([0, 1])
    ax1.arrow(0.08, 1.12, 0.06, 0, head_width=0.01, head_length=0.01, fc='k', ec='k')
    ax1.text(0.08, 1.02, 'moving\nwindow')
    # ax1.grid()

    # draw the phase
    ax2 = ax1.twinx()
    ax2.plot(t, np.angle(z), color='#ff7f0e')
    ax2.tick_params('y', colors='#ff7f0e')
    ax2.set_ylabel('Phase', color='#ff7f0e')
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax2.set_xlim([0, 1])
    ax2.axvline(t[tdamage], ls='--', color='k', lw=2, alpha=0.75)
    ax2.axvspan(t[0], t[nbaseline], alpha=0.4, color='#bcbd22')

    # perform the rolling classification
    pipe = Pipeline([('aug', utils.FeatureAugment()),
                     ('norm', StandardScaler()),
                     ('clf', utils.MultiOCSVM(eta=0.2, kernel='rbf', nu=0.01, gamma=0.2))])

    decision = []
    for i in range(len(z)-nbaseline):
        pipe.fit(z[i:i+nbaseline])
        decision.append(pipe.predict(z[i+nbaseline])[0])

    decision = np.array(decision)
    ind = np.where(decision == 0)[0]

    ax0 = pyplot.subplot(gs[0])
    ax0.plot(t[nbaseline:], decision, '-o', ms=2, color='#2ca02c')
    ax0.plot(t[nbaseline:][ind], decision[ind], 's', ms=4, color='#d62728')
    ax0.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax0.set_xlim([0, 1])
    ax0.set_ylim([-0.1, 1.1])
    ax0.set_xticklabels([])
    ax0.set_yticks([0, 1])
    ax0.set_yticklabels(['damaged', 'healthy'])
    # ax0.grid()
    # pyplot.show()
    pyplot.savefig('analytic_channel_realtime.svg')
