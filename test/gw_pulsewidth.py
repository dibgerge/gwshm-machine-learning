"""
Compute the pulse width of the excitation signal. This is useful to find out the required
parameters for the spectrogram.
"""
from packages import utkit, utils, scihdf

conf = utils.get_configuration()['guided_waves']

info = scihdf.Info(actuator='2T', sensor='5B', impact=0, index=0, frequency=100)

print('Pulse width: ', utils.compute_pw(info)*1e6, 'microsecond')