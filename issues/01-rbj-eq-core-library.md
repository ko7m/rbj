# Core Library: `rbj_eq.py` and `rbj_eq.hpp`

## What These Files Do

These are dual implementations (Python and C++) of the Robert Bristow-Johnson
Audio EQ Cookbook biquad filter formulae. They share an identical architecture
and produce bit-identical results (validated by the golden test suite).

## Coefficient Designers

Eight `design_*()` functions compute normalized biquad coefficients (a0 = 1)
from user-facing parameters:

| Function | Parameters | Notes |
|----------|-----------|-------|
| `design_lpf(f0, Q, Fs)` | Corner freq, Q, sample rate | 2nd-order Butterworth at Q=0.707 |
| `design_hpf(f0, Q, Fs)` | Corner freq, Q, sample rate | |
| `design_bpf_peak(f0, Q, Fs)` | Center freq, Q, sample rate | Constant 0 dB peak gain variant |
| `design_notch(f0, Q, Fs)` | Center freq, Q, sample rate | |
| `design_apf(f0, Q, Fs)` | Center freq, Q, sample rate | |
| `design_peaking(f0, Q, dBgain, Fs)` | + gain in dB | Boost+cut cancel exactly (RBJ Q definition) |
| `design_lowshelf(f0, dBgain, S, Fs)` | + shelf slope S | S=1 gives steepest monotonic slope; `S <= 0` raises |
| `design_highshelf(f0, dBgain, S, Fs)` | + shelf slope S | `S <= 0` raises |

All functions use the bilinear transform with frequency prewarping, exactly as
specified in the Cookbook (`biquadcookbook.txt`).

### Helper Functions

- `A_from_db(dB)` — amplitude factor: `10^(dB/40)` (RBJ's definition, which
  is `sqrt(linear_gain)`)
- `omega0(f0, Fs)` — normalized angular frequency: `2*pi*f0/Fs`
- `alpha_from_Q(w0, Q)` — `sin(w0) / (2*Q)`. Raises (`ValueError` in Python,
  `std::invalid_argument` in C++) if `Q <= 0`, rather than silently producing
  `NaN`/`inf` coefficients that would then poison a `Biquad`'s delay state
  permanently (`NaN` propagates through `z1`/`z2` even across later
  `set_coeffs()` calls with valid values).
- `normalize(b0..a2)` — divides all coefficients by a0

## Biquad Processor

Direct Form II Transposed (DF2T) implementation:

```
y[n] = b0*x[n] + z1
z1   = b1*x[n] - a1*y[n] + z2
z2   = b2*x[n] - a2*y[n]
```

DF2T is preferred for floating-point because it minimizes intermediate signal
range and is less susceptible to numerical issues than DF1 or DF2.

### Python `Biquad` Class

- `process_sample(x)` — single sample
- `process_block(x)` — NumPy array, loop in Python (not vectorized)
- `set_coeffs(b0, b1, b2, a1, a2)` — hot-swap coefficients without reset
- `reset()` — zero the delay state

### C++ `Biquad` Class

- Same interface; `process_block` operates on raw `double*` buffers
- State variables hoisted to locals in the processing loop for performance

## EQBand

Wraps a `Biquad` with its design parameters (kind, f0, Q, gain_db, S, Fs,
enabled). Calls the appropriate `design_*()` function and feeds coefficients
to the internal `Biquad`.

- `update()` — change parameters and redesign on the fly
- `process_block()` — passes through unchanged if `enabled == false`

## ParametricEQ

A serial chain of `EQBand` objects. Processes audio by running each band in
sequence.

- `add_band()` — appends a new EQBand to the chain
- `process_block()` — cascades all bands
- `reset()` — resets all biquad states

## Analysis Helpers (Python Only)

- `freqz_biquad()` — computes complex frequency response H(z) at `worN`
  points from 0 to Nyquist, using direct z-domain evaluation (not FFT)
- `freqz_eq()` — multiplies frequency responses of all bands for total EQ
  response
- `plot_eq_response()` — semilog magnitude plot with optional per-band overlay

## Key Design Decisions

- **No external DSP library**: all filter math is self-contained
- **a0 always normalized to 1**: the `normalize()` step happens once during
  design, not per-sample
- **Shelving filters use S (slope) instead of Q**: this is the RBJ convention
  where S=1 gives the steepest monotonically increasing/decreasing slope
- **Python uses `__slots__`** on `Biquad` for reduced memory overhead
- **C++ is header-only**: `rbj_eq.hpp` can be dropped into any project with
  zero build system changes
- **Q and S are validated at design time**: `Q <= 0` or `S <= 0` raise
  immediately instead of producing `NaN`/`inf` coefficients — a `Biquad`
  fed `NaN` coefficients would latch `NaN` into `z1`/`z2` forever, which
  matters most for `EQBand.update()` in an interactive/live context where a
  bad parameter could otherwise silence a band permanently until `reset()`
