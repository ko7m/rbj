# rbj_eq.py

import numpy as np
import matplotlib

# ---------- helpers ----------

def _A_from_db(dBgain: float) -> float:
    """Amplitude (sqrt of linear gain) from dB gain (RBJ definition)."""
    return 10.0 ** (dBgain / 40.0)

def _omega0(f0: float, Fs: float) -> float:
    return 2.0 * np.pi * f0 / Fs

def _alpha_from_Q(w0: float, Q: float) -> float:
    return np.sin(w0) / (2.0 * Q)

def _normalize(b0, b1, b2, a0, a1, a2):
    """Normalize coefficients so that a0 == 1."""
    b0 /= a0
    b1 /= a0
    b2 /= a0
    a1 /= a0
    a2 /= a0
    return float(b0), float(b1), float(b2), float(a1), float(a2)


# ---------- RBJ coefficient designers (Q-based) ----------

def design_lpf(f0: float, Q: float, Fs: float):
    w0 = _omega0(f0, Fs)
    cosw0 = np.cos(w0)
    alpha = _alpha_from_Q(w0, Q)

    b0 = (1.0 - cosw0) / 2.0
    b1 = 1.0 - cosw0
    b2 = (1.0 - cosw0) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cosw0
    a2 = 1.0 - alpha

    return _normalize(b0, b1, b2, a0, a1, a2)


def design_hpf(f0: float, Q: float, Fs: float):
    w0 = _omega0(f0, Fs)
    cosw0 = np.cos(w0)
    alpha = _alpha_from_Q(w0, Q)

    b0 = (1.0 + cosw0) / 2.0
    b1 = -(1.0 + cosw0)
    b2 = (1.0 + cosw0) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cosw0
    a2 = 1.0 - alpha

    return _normalize(b0, b1, b2, a0, a1, a2)


def design_bpf_peak(f0: float, Q: float, Fs: float):
    """BPF with constant 0 dB peak gain (RBJ's 2nd BPF form)."""
    w0 = _omega0(f0, Fs)
    cosw0 = np.cos(w0)
    alpha = _alpha_from_Q(w0, Q)

    b0 = alpha
    b1 = 0.0
    b2 = -alpha
    a0 = 1.0 + alpha
    a1 = -2.0 * cosw0
    a2 = 1.0 - alpha

    return _normalize(b0, b1, b2, a0, a1, a2)


def design_notch(f0: float, Q: float, Fs: float):
    w0 = _omega0(f0, Fs)
    cosw0 = np.cos(w0)
    alpha = _alpha_from_Q(w0, Q)

    b0 = 1.0
    b1 = -2.0 * cosw0
    b2 = 1.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cosw0
    a2 = 1.0 - alpha

    return _normalize(b0, b1, b2, a0, a1, a2)


def design_apf(f0: float, Q: float, Fs: float):
    w0 = _omega0(f0, Fs)
    cosw0 = np.cos(w0)
    alpha = _alpha_from_Q(w0, Q)

    b0 = 1.0 - alpha
    b1 = -2.0 * cosw0
    b2 = 1.0 + alpha
    a0 = 1.0 + alpha
    a1 = -2.0 * cosw0
    a2 = 1.0 - alpha

    return _normalize(b0, b1, b2, a0, a1, a2)


def design_peaking(f0: float, Q: float, dBgain: float, Fs: float):
    """RBJ peaking EQ with his Q definition (boost+cut cancels)."""
    A = _A_from_db(dBgain)
    w0 = _omega0(f0, Fs)
    cosw0 = np.cos(w0)
    alpha = _alpha_from_Q(w0, Q)

    b0 = 1.0 + alpha * A
    b1 = -2.0 * cosw0
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cosw0
    a2 = 1.0 - alpha / A

    return _normalize(b0, b1, b2, a0, a1, a2)


def design_lowshelf(f0: float, dBgain: float, S: float, Fs: float):
    A = _A_from_db(dBgain)
    w0 = _omega0(f0, Fs)
    cosw0 = np.cos(w0)
    sinw0 = np.sin(w0)
    alpha = sinw0 / 2.0 * np.sqrt((A + 1.0 / A) * (1.0 / S - 1.0) + 2.0)

    two_sqrtA_alpha = 2.0 * np.sqrt(A) * alpha

    b0 = A * ((A + 1.0) - (A - 1.0) * cosw0 + two_sqrtA_alpha)
    b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cosw0)
    b2 = A * ((A + 1.0) - (A - 1.0) * cosw0 - two_sqrtA_alpha)
    a0 = (A + 1.0) + (A - 1.0) * cosw0 + two_sqrtA_alpha
    a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cosw0)
    a2 = (A + 1.0) + (A - 1.0) * cosw0 - two_sqrtA_alpha

    return _normalize(b0, b1, b2, a0, a1, a2)


def design_highshelf(f0: float, dBgain: float, S: float, Fs: float):
    A = _A_from_db(dBgain)
    w0 = _omega0(f0, Fs)
    cosw0 = np.cos(w0)
    sinw0 = np.sin(w0)
    alpha = sinw0 / 2.0 * np.sqrt((A + 1.0 / A) * (1.0 / S - 1.0) + 2.0)

    two_sqrtA_alpha = 2.0 * np.sqrt(A) * alpha

    b0 = A * ((A + 1.0) + (A - 1.0) * cosw0 + two_sqrtA_alpha)
    b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cosw0)
    b2 = A * ((A + 1.0) + (A - 1.0) * cosw0 - two_sqrtA_alpha)
    a0 = (A + 1.0) - (A - 1.0) * cosw0 + two_sqrtA_alpha
    a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cosw0)
    a2 = (A + 1.0) - (A - 1.0) * cosw0 - two_sqrtA_alpha

    return _normalize(b0, b1, b2, a0, a1, a2)


# ---------- DF2T biquad ----------

class Biquad:
    """
    DF2T biquad:
        y[n] = b0*x[n] + z1
        z1'  = b1*x[n] - a1*y[n] + z2
        z2'  = b2*x[n] - a2*y[n]
    """
    __slots__ = ("b0", "b1", "b2", "a1", "a2", "z1", "z2")

    def __init__(self, b0, b1, b2, a1, a2):
        self.b0 = float(b0)
        self.b1 = float(b1)
        self.b2 = float(b2)
        self.a1 = float(a1)
        self.a2 = float(a2)
        self.z1 = 0.0
        self.z2 = 0.0

    def reset(self):
        self.z1 = 0.0
        self.z2 = 0.0

    def process_sample(self, x: float) -> float:
        y = self.b0 * x + self.z1
        self.z1 = self.b1 * x - self.a1 * y + self.z2
        self.z2 = self.b2 * x - self.a2 * y
        return y

    def process_block(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.empty_like(x)
        z1 = self.z1
        z2 = self.z2
        b0, b1, b2, a1, a2 = self.b0, self.b1, self.b2, self.a1, self.a2

        for i in range(len(x)):
            xi = x[i]
            yi = b0 * xi + z1
            z1 = b1 * xi - a1 * yi + z2
            z2 = b2 * xi - a2 * yi
            y[i] = yi

        self.z1 = z1
        self.z2 = z2
        return y

    def set_coeffs(self, b0, b1, b2, a1, a2):
        self.b0 = float(b0)
        self.b1 = float(b1)
        self.b2 = float(b2)
        self.a1 = float(a1)
        self.a2 = float(a2)


# ---------- EQ bands + chain ----------

class EQBand:
    def __init__(self, kind: str, f0: float, Q: float = 1.0,
                 gain_db: float = 0.0, S: float = 1.0,
                 Fs: float = 48000.0, enabled: bool = True):
        self.kind = kind      # "lpf", "hpf", "peaking", "lowshelf", ...
        self.f0 = f0
        self.Q = Q
        self.gain_db = gain_db
        self.S = S
        self.Fs = Fs
        self.enabled = enabled
        self.biquad = None
        self._update_biquad()

    def _design(self):
        k = self.kind.lower()
        if k == "lpf":
            return design_lpf(self.f0, self.Q, self.Fs)
        elif k == "hpf":
            return design_hpf(self.f0, self.Q, self.Fs)
        elif k == "bpf":
            return design_bpf_peak(self.f0, self.Q, self.Fs)
        elif k == "notch":
            return design_notch(self.f0, self.Q, self.Fs)
        elif k == "apf":
            return design_apf(self.f0, self.Q, self.Fs)
        elif k == "peaking":
            return design_peaking(self.f0, self.Q, self.gain_db, self.Fs)
        elif k == "lowshelf":
            return design_lowshelf(self.f0, self.gain_db, self.S, self.Fs)
        elif k == "highshelf":
            return design_highshelf(self.f0, self.gain_db, self.S, self.Fs)
        else:
            raise ValueError(f"Unknown EQ band kind: {self.kind}")

    def _update_biquad(self):
        b0, b1, b2, a1, a2 = self._design()
        if self.biquad is None:
            self.biquad = Biquad(b0, b1, b2, a1, a2)
        else:
            self.biquad.set_coeffs(b0, b1, b2, a1, a2)

    def update(self, **kwargs):
        """Update band parameters and redesign the filter."""
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._update_biquad()

    def process_block(self, x: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return x
        return self.biquad.process_block(x)


class ParametricEQ:
    def __init__(self, Fs: float = 48000.0):
        self.Fs = Fs
        self.bands = []

    def add_band(self, **kwargs) -> EQBand:
        kwargs.setdefault("Fs", self.Fs)
        band = EQBand(**kwargs)
        self.bands.append(band)
        return band

    def process_block(self, x: np.ndarray) -> np.ndarray:
        y = x
        for band in self.bands:
            y = band.process_block(y)
        return y

    def reset(self):
        for band in self.bands:
            band.biquad.reset()

# ---------- analysis / plotting helpers ----------

def freqz_biquad(b0, b1, b2, a1, a2, Fs=48000.0, worN=2048):
    """
    Compute frequency response of a single normalized biquad (a0=1).
    Returns (freqs_Hz, H_complex).
    """
    w = np.linspace(0.0, np.pi, worN)
    ejw = np.exp(1j * w)
    z1 = 1.0 / ejw
    z2 = z1**2

    num = b0 + b1 * z1 + b2 * z2
    den = 1.0 + a1 * z1 + a2 * z2

    H = num / den
    freqs = w * Fs / (2.0 * np.pi)
    return freqs, H


def freqz_eq(eq: ParametricEQ, Fs=48000.0, worN=2048):
    """
    Combined frequency response of all bands in a ParametricEQ.
    Returns (freqs_Hz, H_total_complex).
    """
    w = np.linspace(0.0, np.pi, worN)
    H_total = np.ones_like(w, dtype=np.complex128)

    for band in eq.bands:
        b = band.biquad
        freqs, H = freqz_biquad(b.b0, b.b1, b.b2, b.a1, b.a2, Fs=Fs, worN=worN)
        H_total *= H  # cascade = product of transfer functions

    return freqs, H_total


def plot_eq_response(eq: ParametricEQ, Fs=48000.0, worN=2048,
                     show_bands=False):
    """
    Plot magnitude response of the whole EQ.
    If show_bands=True, also plot each band's response faintly.
    """
    import matplotlib.pyplot as plt

    freqs, H_total = freqz_eq(eq, Fs=Fs, worN=worN)
    mag_total = 20.0 * np.log10(np.maximum(np.abs(H_total), 1e-12))

    plt.figure()
    plt.semilogx(freqs, mag_total, label="Total EQ")

    if show_bands:
        for i, band in enumerate(eq.bands):
            b = band.biquad
            f, H = freqz_biquad(b.b0, b.b1, b.b2, b.a1, b.a2, Fs=Fs, worN=worN)
            mag = 20.0 * np.log10(np.maximum(np.abs(H), 1e-12))
            plt.semilogx(f, mag, linestyle="--", alpha=0.4,
                         label=f"Band {i+1}: {band.kind}")

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Parametric EQ Magnitude Response")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.xlim([20, Fs/2])
    plt.show()

