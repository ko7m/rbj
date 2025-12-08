# live_eq.py

import numpy as np
import sounddevice as sd

from rbj_eq import ParametricEQ

Fs = 48000
blocksize = 256    # adjust for your latency/CPU tradeoff
channels = 2       # or 2 for stereo

# ----- build EQ -----
eqL = ParametricEQ(Fs=Fs)
# example bands
eqL.add_band(kind="lowshelf", f0=100.0,  gain_db=6.0,  S=1.0)   # bass boost
eqL.add_band(kind="peaking",  f0=2000.0, Q=1.0,        gain_db=-3.0)
eqL.add_band(kind="highshelf", f0=8000.0, gain_db=4.0, S=0.7)

# for stereo, clone bands into a second EQ
eqR = None
if channels == 2:
    eqR = ParametricEQ(Fs=Fs)
    for b in eqL.bands:
        eqR.add_band(kind=b.kind, f0=b.f0, Q=b.Q,
                     gain_db=b.gain_db, S=b.S)


def audio_callback(indata, outdata, frames, time, status):
    if status:
        # prints XRuns or other driver warnings
        print(status)

    # indata/outdata: shape (frames, channels)
    x = indata.copy()

    if channels == 1:
        y = eqL.process_block(x[:, 0])
        outdata[:, 0] = y.astype(outdata.dtype)
    else:
        # process L and R independently
        yL = eqL.process_block(x[:, 0])
        yR = eqR.process_block(x[:, 1])
        outdata[:, 0] = yL.astype(outdata.dtype)
        outdata[:, 1] = yR.astype(outdata.dtype)


if __name__ == "__main__":
    with sd.Stream(channels=channels,
                   samplerate=Fs,
                   blocksize=blocksize,
                   dtype='float32',
                   callback=audio_callback):
        print("Live EQ running. Ctrl+C to stop.")
        try:
            while True:
                pass
        except KeyboardInterrupt:
            print("\nStopping.")
