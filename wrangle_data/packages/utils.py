from os.path import join, abspath, dirname
import numpy as np
from . import utkit, scihdf
import yaml


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
    project_dir = join(code_dir, '..', '..')

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
        return join(self.project_dir, *seq)
