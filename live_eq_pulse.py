import numpy as np
import sounddevice as sd
from rbj_eq import ParametricEQ, plot_eq_response

Fs = 48000
blocksize = 256
channels = 2    # stereo in, stereo out for Pulse
eqActive = 0

print("=== Available devices ===")
print(sd.query_devices())
print("Default device:", sd.default.device)
print("=========================\n")

# ----- build EQ for L and R -----
eqL = ParametricEQ(Fs=Fs)
if eqActive:
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

eqR = ParametricEQ(Fs=Fs)
for b in eqL.bands:
    eqR.add_band(kind=b.kind, f0=b.f0, Q=b.Q,
                 gain_db=b.gain_db, S=b.S)

def audio_callback(indata, outdata, frames, time, status):
    if status:
        print(status)

    # Stereo in, stereo out
    xL = indata[:, 0]
    xR = indata[:, 1]

    yL = eqL.process_block(xL)
    yR = eqR.process_block(xR)

    outdata[:, 0] = yL
    outdata[:, 1] = yR

if __name__ == "__main__":
    # Use Pulse by NAME, not index
    device = ('pulse', 'pulse')

    print("Starting Pulse EQ...")
    with sd.Stream(
        device=device,         # (input, output)
        samplerate=Fs,
        blocksize=blocksize,
        dtype='float32',
        channels=channels,     # 2 in, 2 out
        callback=audio_callback,
    ):
        print("EQ is running. Ctrl+C to stop.")
        try:
            while True:
                sd.sleep(100)
        except KeyboardInterrupt:
            print("\nStopping.")
