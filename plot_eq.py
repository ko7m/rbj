import numpy as np
import sounddevice as sd
from rbj_eq import ParametricEQ, plot_eq_response

Fs = 48000
blocksize = 256
channels = 2    # stereo in, stereo out for Pulse

print("=== Available devices ===")
print(sd.query_devices())
print("Default device:", sd.default.device)
print("=========================\n")

# ----- build EQ for L and R -----
eqL = ParametricEQ(Fs=Fs)
#eqL.add_band(kind="lowshelf",  f0=100.0,  gain_db=6.0,  S=1.0)
#eqL.add_band(kind="peaking",   f0=2000.0, Q=1.0,        gain_db=-3.0)
#eqL.add_band(kind="highshelf", f0=8000.0, gain_db=4.0,  S=0.7)

# boomy
#eqL.add_band(kind="lowshelf",  f0=200.0,  gain_db=12.0, S=1.0)   # big bass boost
#eqL.add_band(kind="peaking",   f0=2000.0, Q=0.5,        gain_db=-15.0)
#eqL.add_band(kind="highshelf", f0=4000.0, gain_db=-12.0, S=1.0)  # kill highs

# thin and bright
eqL.add_band(kind="highshelf", f0=4000.0, gain_db=15.0, S=1.0)   # sparkle boost
eqL.add_band(kind="lowshelf",  f0=200.0,  gain_db=-12.0, S=1.0)  # remove bass

plot_eq_response(eqL, Fs=Fs, worN=4096, show_bands=True)

