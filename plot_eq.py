import numpy as np
import sounddevice as sd
from rbj_eq import ParametricEQ, plot_eq_response

Fs = 48000
blocksize = 256
channels = 2    # stereo in, stereo out for Pulse

# ----- build EQ for L and R -----

# Pass-thru
eqL = ParametricEQ(Fs=Fs)

# Rough “SSB-ish” audio shaping
#eqL.add_band(kind="highshelf", f0=300.0,  gain_db=+6,  S=0.7)   # bring up speech fundamentals
#eqL.add_band(kind="lowshelf",  f0=2500.0, gain_db=-12, S=0.7)   # roll off highs

# Nasty carrier at, say, 1.2 kHz audio
#eqL.add_band(kind="notch", f0=1200.0, Q=10.0)

#
#eqL.add_band(kind="lowshelf",  f0=100.0,  gain_db=6.0,  S=1.0)
#eqL.add_band(kind="peaking",   f0=2000.0, Q=1.0,        gain_db=-3.0)
#eqL.add_band(kind="highshelf", f0=8000.0, gain_db=4.0,  S=0.7)

# boomy
#eqL.add_band(kind="lowshelf",  f0=200.0,  gain_db=12.0, S=1.0)   # big bass boost
#eqL.add_band(kind="peaking",   f0=2000.0, Q=0.5,        gain_db=-15.0)
#eqL.add_band(kind="highshelf", f0=4000.0, gain_db=-12.0, S=1.0)  # kill highs

# thin and bright
#eqL.add_band(kind="highshelf", f0=4000.0, gain_db=15.0, S=1.0)   # sparkle boost
#eqL.add_band(kind="lowshelf",  f0=200.0,  gain_db=-12.0, S=1.0)  # remove bass

# Stacked notch filters
#eqL.add_band(kind="notch", f0=1000, Q=10)
#eqL.add_band(kind="notch", f0=1500, Q=10)
#eqL.add_band(kind="notch", f0=700,  Q=20)

# Default Hi-Fi response limits
#plot_eq_response(eqL, Fs=Fs, worN=4096, show_bands=True)

# 1kHz notch filter, 100Hz at -3dB  Q = 1000/100 == 10
#eqL.add_band(kind="notch", f0=1000.0, Q=10.0)
#plot_eq_response(eqL, Fs=Fs, worN=4096, show_bands=True, xlim=(400, 2000), ylim=(-40, 5))

# 170 Hz FSK filter centred around 850 Hz
f0 = 850.0          # Hz
BW = 250.0          # desired -3 dB width
Q  = f0 / BW        # ≈ 3.4
eqL.add_band(kind="bpf", f0=f0, Q=Q)
plot_eq_response(eqL, Fs=Fs, worN=4096, show_bands=True, xlim=(500, 1300), ylim=(-40, 5))

