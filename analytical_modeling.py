import numpy as np
import pandas as pd
from matplotlib import pyplot


nx, ny = 3, 3

baselines = np.exp(1j*np.linspace(-np.pi/5, np.pi/5))

baseline_cloud = baselines - baselines.reshape(-1, 1)

#nodefect = np.exp(1j*np.linspace(-np.pi/4, np.pi/4))
nodefect = np.exp(1j*np.pi*2)*0.78
nodefect_cloud = nodefect - baselines.reshape(-1, 1)

fig, ax = pyplot.subplots(nx, ny)
fig2 = pyplot.figure()
ax2 = pyplot.gca()

for i in range(nx):
    for j in range(ny):
        # make new amplitude of defect everytime
        A = 0.2 + 0.2*(i*nx + j)
        defect = nodefect + np.exp(1j * np.pi/4) * A
        defect_cloud = defect - baselines.reshape(-1, 1)

        ax[j, i].plot(np.real(baseline_cloud), np.imag(baseline_cloud), 'o', color='#1f77b4')
        ax[j, i].plot(np.real(defect_cloud), np.imag(defect_cloud), '-s', color='#ff7f0e')
        ax[j, i].plot(np.real(nodefect_cloud), np.imag(nodefect_cloud), '-x', color='#e377c2')
        ax[j, i].set_title(str(A))
        ax[j, i].grid()
        ax[j, i].set_aspect('equal')

        ax2.plot((i*nx + j)*np.ones(baseline_cloud.reshape(-1).shape),
                 abs(baseline_cloud.reshape(-1)), 'o', color='#1f77b4')
        ax2.plot((i*nx + j)*np.ones(defect_cloud.reshape(-1).shape),
                 abs(defect_cloud.reshape(-1)),
                 's', color='#ff7f0e')

pyplot.show()

