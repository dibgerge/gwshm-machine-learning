"""
computes the subtraction of each feature from the baseline (no impact) data. That is, for each
sensor, subtract the values at impact 0 for all the other values at all impacts.

Note: Very slow, can be significantly improved...
"""
from packages import utils
import pandas as pd

conf = utils.Configurations()['journal_2017']['features']['complex_amp']

feat = pd.read_excel(conf['dump'], index_col=(0, 1, 2))
feat = feat.applymap(complex)
baseline = feat.loc[pd.IndexSlice[:, 0, :], :]

mux = pd.MultiIndex(levels=[[], [], [], []], labels=[[], [], [], []],
                    names=['sensor', 'impact', 'baseline_index', 'index'])

out = pd.DataFrame(0, index=mux, columns=feat.columns)

for (sensor, _, baseline_index), b in baseline.iterrows():
    print('Sensor:', sensor, 'Baseline index:', baseline_index)
    for (_, impact, index), s in feat.loc[pd.IndexSlice[sensor, :, :], :].iterrows():
        out.loc[pd.IndexSlice[sensor, impact, baseline_index, index]] = s - b

# write to excel sheet
out.sort_index(axis=0, inplace=True)
writer = pd.ExcelWriter(conf['dump_baselined'])
out.astype(str).to_excel(writer)
writer.save()
writer.close()
