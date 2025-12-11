#!/usr/bin/env python3
import numpy as np
import rbj_eq  # your module above

# A small set of test cases; add more as desired.
TESTS = [
    # kind, f0,   Q,      gain_db,  S,    Fs
    ("lpf",      1000.0, 0.707,   0.0,    1.0, 48000.0),
    ("hpf",      200.0,  0.5,     0.0,    1.0, 48000.0),
    ("bpf",      3000.0, 2.0,     0.0,    1.0, 48000.0),
    ("notch",    1000.0, 1.0,     0.0,    1.0, 48000.0),
    ("apf",      500.0,  0.7,     0.0,    1.0, 48000.0),
    ("peaking",  4000.0, 1.0,     6.0,    1.0, 48000.0),
    ("peaking",  4000.0, 1.0,    -6.0,    1.0, 48000.0),
    ("lowshelf", 200.0,  0.707,   6.0,    1.0, 48000.0),
    ("highshelf",8000.0, 0.707,  -3.0,    1.0, 48000.0),
]

IMPULSE_LEN = 64  # long enough to see transient behavior

def design_band(kind, f0, Q, gain_db, S, Fs):
    k = kind.lower()
    if k == "lpf":
        return rbj_eq.design_lpf(f0, Q, Fs)
    elif k == "hpf":
        return rbj_eq.design_hpf(f0, Q, Fs)
    elif k == "bpf":
        return rbj_eq.design_bpf_peak(f0, Q, Fs)
    elif k == "notch":
        return rbj_eq.design_notch(f0, Q, Fs)
    elif k == "apf":
        return rbj_eq.design_apf(f0, Q, Fs)
    elif k == "peaking":
        return rbj_eq.design_peaking(f0, Q, gain_db, Fs)
    elif k == "lowshelf":
        return rbj_eq.design_lowshelf(f0, gain_db, S, Fs)
    elif k == "highshelf":
        return rbj_eq.design_highshelf(f0, gain_db, S, Fs)
    else:
        raise ValueError(f"Unknown kind: {kind}")

def main():
    with open("rbj_golden.txt", "w") as f:
        for idx, (kind, f0, Q, gain_db, S, Fs) in enumerate(TESTS):
            # Design coefficients
            b0, b1, b2, a1, a2 = design_band(kind, f0, Q, gain_db, S, Fs)

            # Impulse response via your Python Biquad
            bq = rbj_eq.Biquad(b0, b1, b2, a1, a2)
            x = np.zeros(IMPULSE_LEN, dtype=np.float64)
            x[0] = 1.0
            y = bq.process_block(x)

            f.write(f"# test {idx}\n")
            f.write(
                f"{kind} {f0:.10g} {Q:.10g} {gain_db:.10g} {S:.10g} {Fs:.10g} {IMPULSE_LEN}\n"
            )
            f.write("coeffs %.17g %.17g %.17g %.17g %.17g\n" % (b0, b1, b2, a1, a2))
            f.write("impulse " + " ".join("%.17g" for _ in range(IMPULSE_LEN)) % tuple(y) + "\n\n")

if __name__ == "__main__":
    main()
