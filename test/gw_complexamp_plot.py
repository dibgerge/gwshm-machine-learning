"""
Show the baselined complex amplitude features in a 2-D space plot.
"""
from packages import utils
import numpy as np
import pandas as pd
from matplotlib import pyplot

conf = utils.get_configuration()['guided_waves']['features']['complex_amp']
feat = pd.read_excel(conf['dump_baselined'], index_col=(0, 1, 2, 3)).applymap(complex)

subset = feat.loc[pd.IndexSlice['5T', 0, :, :]]
pyplot.plot(subset.real, subset.imag, 'o', ms=8)
pyplot.show()
