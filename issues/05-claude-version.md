# Alternative Implementation: `claude_version.py`

## What This File Is

A self-contained reimplementation of the RBJ biquad filter cookbook, generated
by Claude (AI). It takes a different architectural approach from the original
`rbj_eq.py` while implementing the same underlying math.

## Key Differences from `rbj_eq.py`

| Aspect | `rbj_eq.py` | `claude_version.py` |
|--------|-------------|---------------------|
| Architecture | Separate `design_*()` functions + `Biquad` + `EQBand` + `ParametricEQ` | Single `BiquadFilter` class does everything |
| Filter selection | String-based (`"lpf"`, `"peaking"`, etc.) | `FilterType` enum |
| Processing form | Direct Form II Transposed (DF2T) | Direct Form 1 (DF1) |
| Alpha source | Q only (shelves use S) | Q, bandwidth (octaves), or shelf slope — all three supported |
| BPF variants | Only constant 0 dB peak gain | Both constant-skirt-gain and constant-peak-gain |
| State variables | `z1`, `z2` (DF2T delay line) | `x1`, `x2`, `y1`, `y2` (DF1 delay line) |
| Block processing | `process_block()` | `process()` |
| Freq response | `freqz_biquad()` free function | `get_frequency_response()` method on the filter |
| Multi-band EQ | `ParametricEQ` class chains bands | Not provided |
| A calculation | `10^(dB/40)` — RBJ's definition | `sqrt(10^(dB/20))` — mathematically identical |
| Default Fs | 48000 Hz | 44100 Hz (in the demo code) |
| Plotting | matplotlib semilog plot | Frequency response data only (no plot) |

## `FilterType` Enum

```python
LPF         = "lowpass"
HPF         = "highpass"
BPF_SKIRT   = "bandpass_skirt"   # constant skirt gain, peak = Q
BPF_PEAK    = "bandpass_peak"    # constant 0 dB peak gain
NOTCH       = "notch"
APF         = "allpass"
PEAKING_EQ  = "peaking"
LOW_SHELF   = "lowshelf"
HIGH_SHELF  = "highshelf"
```

Nine filter types vs eight in `rbj_eq.py` — the additional one is `BPF_SKIRT`,
which uses `b0 = sin(w0)/2` instead of `b0 = alpha`. This is the first BPF
form in the RBJ Cookbook (constant skirt gain, peak gain = Q).

## `BiquadFilter` Class

### Constructor

```python
BiquadFilter(filter_type, sample_rate, frequency,
             q=0.707, db_gain=0.0, bandwidth=None, shelf_slope=None)
```

Calculates all six raw coefficients in `_calculate_coefficients()`, then
normalizes by a0 in the constructor. Alpha is selected based on which
optional parameter is provided:

1. `bandwidth` → `alpha = sin(w0) * sinh(ln(2)/2 * BW * w0/sin(w0))`
2. `shelf_slope` → `alpha = sin(w0)/2 * sqrt((A + 1/A)*(1/S - 1) + 2)`
3. Otherwise → `alpha = sin(w0) / (2*Q)` (default)

### Processing (Direct Form 1)

```
y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
```

DF1 stores both input and output history (4 state variables) vs DF2T's
2 state variables. DF1 is the "most straightforward" implementation per the
Cookbook text itself, and is better suited to fixed-point. DF2T (used in
`rbj_eq.py`) is generally preferred for floating-point.

### Frequency Response

`get_frequency_response(frequencies=None)` evaluates H(z) directly in the
z-domain and returns `(frequencies, magnitude_dB, phase_degrees)`. Uses
log-spaced frequencies from 1 Hz to Nyquist by default. Includes a small
epsilon (`1e-10`) in the dB calculation to avoid log-of-zero.

### Demo / Self-Test

The `if __name__ == "__main__"` block:
- Creates a 1-second test signal at 44.1 kHz with 100 Hz, 1 kHz, and 5 kHz
  sine components
- Runs it through LPF (2 kHz), HPF (500 Hz), and peaking EQ (+6 dB at 1 kHz)
- Prints filter descriptions and the LPF magnitude at 1 kHz

## Relationship to `rbj_eq.py`

Both implement the same cookbook formulae and should produce identical
coefficients for equivalent parameters. The main value of `claude_version.py`
is as a comparison point — different structure, different processing form,
same underlying math. It could also serve as a starting point for a more
"Pythonic" API with enums and type hints, though it lacks the multi-band EQ
chain that makes `rbj_eq.py` useful for the live audio applications.
