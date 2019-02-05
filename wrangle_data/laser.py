""" Reads the raw .mat optical laser datafiles and stores them in an HDF5 file."""
from packages import utils, scihdf
import pandas as pd
import numpy as np
from glob import glob
from os.path import join, basename
from scipy.io import loadmat


conf = utils.Configurations()


def read_form_filename(filename):
    """
    Extract measurement information for a given laser measurement from its filename.

    Parameters
    ----------
    filename : str
        Filename of the laser measurement.

    Returns
    -------
    info : dict
        A dictionary containing the laser measurement parameters.
    """
    # if fullpath is provided, extract the filename only
    filename = basename(filename)

    impact_energy = float(filename[:4])
    parts = filename.split('_')

    if len(parts) == 2:
        impact_num = 0
    elif len(parts) == 3:
        impact_num = int(parts[1][0]) - 1
    else:
        raise ValueError('The filename provided does not follow required naming convention.')

    impact_idx = conf['guided_waves']['experiment']['impact_energy'].index(impact_energy) + \
                 impact_num
    return scihdf.Info(impact=impact_idx)


# Open the default laser HDF5 storage database
store = scihdf.SciHDF(conf['laser']['dump'], mode='a')
fnames = glob(join(conf['laser']['root_dir'], '*.mat'))

for fname in fnames:
    info = read_form_filename(fname)
    print(info)
    d = loadmat(fname)['Data'][0]
    store[info] = pd.DataFrame(d[2], index=d[1][:, 0], columns=d[0][0, :], dtype=np.float32)

store.close()
