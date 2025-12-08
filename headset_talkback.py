# live_eq.py
import numpy as np
import sounddevice as sd

from rbj_eq import ParametricEQ

Fs = 48000
blocksize = 256

# ----- build EQ (mono) -----
eq = ParametricEQ(Fs=Fs)
#eq.add_band(kind="lowshelf",  f0=100.0,  gain_db=6.0,  S=1.0)
#eq.add_band(kind="peaking",   f0=2000.0, Q=1.0,        gain_db=-3.0)
#eq.add_band(kind="highshelf", f0=8000.0, gain_db=4.0,  S=0.7)

# Massive bass boost and kill the highs
#eq.add_band(kind="lowshelf",  f0=200.0,  gain_db=12.0, S=1.0)     # huge bass boost
#eq.add_band(kind="highshelf", f0=4000.0, gain_db=-12.0, S=1.0)    # kill highs



def audio_callback(indata, outdata, frames, time, status):
    if status:
        print(status)

    # Because channels=(1, 2):
    #   indata  shape = (frames, 1)
    #   outdata shape = (frames, 2)

    x = indata[:, 0]          # mono input (headset mic, or whatever the source is)
    y = eq.process_block(x)   # process in mono

    # send mono to both ears
    outdata[:, 0] = y
    outdata[:, 1] = y


if __name__ == "__main__":
    device = 0  # Logitech USB Headset: Audio (1 in, 2 out)

    with sd.Stream(
        device=device,
        samplerate=Fs,
        blocksize=blocksize,
        dtype='float32',
        channels=(1, 2),   # <-- 1 input channel, 2 output channels
        callback=audio_callback,
    ):
        print("Live EQ on Logitech headset (mic → phones). Ctrl+C to stop.")
        try:
            while True:
                sd.sleep(100)  # sleep 100 ms and let the callback do its thing
        except KeyboardInterrupt:
            print("\nStopping.")
