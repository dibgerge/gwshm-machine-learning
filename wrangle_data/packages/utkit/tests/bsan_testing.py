import utkit
from utkit.io import civa_bscan
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from time import time


def test_interp2():
    probe_delay = 2 * 25 * 1e-3 / 2680.0
    bscan = civa_bscan(join('..', 'data', 'B04_E_5MHz_60S_+x_HalfInch_bscan.txt'))
    bscan = bscan.shift_axes(probe_delay,
                             axis=0).scale_axes(np.cos(np.deg2rad(60)) * 3156.0 / 2.0, axes=0).operate('nde')
    bscan = bscan.iloc[::30,:].copy()
    ct = time()
    bscan = bscan.skew(60, axes=0, ts=[0.05e-3, 0.05e-3], method='cubic')
    print(time() - ct)
    #print(bscan.head())
    bscan = bscan.fillna(10)
    plt.figure()
    plt.pcolormesh(bscan.columns, bscan.index, bscan.values, vmax=0, vmin=-50)
    ax = plt.gca()
    ax.invert_yaxis()
    ax.set_aspect('equal')


def interp(option='linear'):
    probe_delay = 2*25*1e-3/2680.0

    # new grid to compute
    gridx, gridy = np.mgrid[0.02:0.080:50j, 0.00:0.030:50j]

    bscan = civa_bscan(join('..', 'data', 'B04_E_5MHz_60S_+x_HalfInch_bscan.txt'))
    bscan = bscan.shift_axes(probe_delay, axis=0).scale_axes(3156.0 / 2.0, axes=0).operate('nde')
    bscan = bscan.iloc[::20, :].copy()
    X, Y = np.meshgrid(bscan.columns, bscan.index, indexing='xy')
    X += Y * np.sin(np.deg2rad(60))
    Y *= np.cos(np.deg2rad(60))

    # plt.figure()
    # plt.pcolormesh(X, Y, bscan.values, vmax=0, vmin=-50)
    # ax = plt.gca()
    # ax.invert_yaxis()
    # ax.set_aspect('equal')

    ct = time()
    # perform the interpolation
    if option == 'linear':
        from scipy.interpolate import LinearNDInterpolator
        f = LinearNDInterpolator((Y.ravel(), X.ravel()), bscan.values.ravel(), fill_value=10)
        gridx, gridy = gridx.T, gridy.T
        vals = f(gridy, gridx)
    elif option == 'nearest':
        from scipy.interpolate import NearestNDInterpolator
        f = NearestNDInterpolator((Y.ravel(), X.ravel()), bscan.values.ravel())
        gridx, gridy = gridx.T, gridy.T
        vals = f(gridy, gridx)
    elif option == 'griddata':
        from scipy.interpolate import griddata
        vals = griddata((Y.ravel(), X.ravel()), bscan.values.ravel(),
                        (gridy.ravel(), gridx.ravel()), method='linear', fill_value=10)
        #gridy, gridx = gridy.ravel(), gridx.ravel()
    print('Total time: ', time() - ct)

    plt.figure()
    plt.pcolormesh(gridx, gridy, vals.reshape(gridx.shape), vmax=0, vmin=-50, edgecolor='k')
    ax = plt.gca()
    ax.invert_yaxis()
    ax.set_aspect('equal')



def interp_spline():
    bscan = civa_bscan(join('..', 'data', 'B02_K_5MHz_60S_-x_HalfInch_bscan.txt'))
    gridx, gridy = np.mgrid[20e-6:50e-6:50j, 0.:30e-3:50j]
    from scipy.interpolate import griddata
    s = bscan.stack()
    vals = griddata(np.array(list(zip(*s.index.values))).T, s.values, (gridy, gridx))
    plt.imshow(vals)
    plt.show()


if __name__ == '__main__':
    test_interp2()
    # interp('nearest')
    # interp('linear')
    # interp('griddata')
    plt.show()