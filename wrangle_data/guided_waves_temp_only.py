"""
Saves the guided wave data which has been varied with temperature.
"""
import numpy as np
from packages import utils, scihdf
from glob import glob
from os.path import join, basename
from scipy.io import loadmat
import pandas as pd

conf = utils.Configurations()['guided_waves']


def _get_info_from_file(filename):
    """
    Extracts measurement information for a given guided wave measurement file from its filename.
    Parameters
    ----------
    filename : str
        Filename of the guided wave measurement.
    """
    # if fullpath is provided, extract the filename only
    filename = basename(filename)

    osc = int(filename[3])

    ind = filename.lower().find('khz')
    frequency = int(filename[5:ind])
    actuator = filename[ind+3:ind+5]

    impact_energy = float(filename[ind+5:ind+9])
    parts = filename.split('_')
    impact_num = int(parts[1][0]) - 1 if len(parts) == 2 else 0

    impact_idx = conf['experiment']['impact_energy'].index(impact_energy) + impact_num

    ind = -filename[::-1].index('V')
    parts = filename[ind:].split('.')
    temperature_ind = int(parts[0]) if parts[0] else 0

    if frequency != 100 or actuator != '2T':
        return None, None

    return osc, scihdf.Info(sensor=None,
                            impact=impact_idx,
                            index=temperature_ind)


def main():
    store = scihdf.SciHDF(conf['data']['rx_dump'], mode='a', complib='zlib')
    store_excite = scihdf.SciHDF(conf['data']['tx_dump'], mode='a', complib='zlib')

    fnames = glob(join(conf['data']['raw_dir'], '*.mat'))
    # loop over files containing different temperatures
    for fn in fnames:
        osc, info = _get_info_from_file(fn)

        if osc is None:
            continue

        d = loadmat(fn)['Data']
        act_sig = pd.Series(d[:, 1], index=d[:, 0], dtype=np.float32)

        for i, sig in enumerate(d[:, 2:].T):
            sensor_id = conf['experiment']['sensor_id'][3*(osc - 1)+i]
            info.modify(sensor=sensor_id)
            print(info)
            sensor_sig = pd.Series(sig, index=d[:, 0], dtype=np.float32)
            store[info] = sensor_sig
            store_excite[info] = act_sig
    store.close()
    store_excite.close()


if __name__ == '__main__':
    main()
