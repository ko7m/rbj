# Project Overview: rbj

## What It Is

An implementation of Robert Bristow-Johnson's (RBJ) "Audio EQ Cookbook" biquad
filter formulae in both **Python** and **C++**, with live real-time audio
processing applications built on top.

The RBJ Cookbook is the canonical reference for computing biquad IIR filter
coefficients for audio equalization. This project provides:

1. A **Python library** (`rbj_eq.py`) — coefficient designers, a DF2T biquad
   processor, a multi-band parametric EQ chain, and frequency-response plotting
   helpers.
2. A **C++ header-only library** (`rbj_eq.hpp`) — a port of the same
   coefficient designers and DF2T biquad/EQ chain, suitable for embedding in
   real-time audio pipelines.
3. Several **live audio scripts** that apply EQ in real-time to a sound device
   (USB headset or PulseAudio) using `sounddevice` (PortAudio).
4. A **cross-language golden test** harness: Python generates reference
   coefficients and impulse responses (`gen_golden.py` → `rbj_golden.txt`),
   and a C++ test (`test_rbj.cpp`) validates the C++ port against them.

## Supported Filter Types

All eight standard RBJ biquad types are implemented:

| Filter | Description |
|--------|-------------|
| LPF | Low-pass filter |
| HPF | High-pass filter |
| BPF | Band-pass filter (constant 0 dB peak gain) |
| Notch | Band-reject / notch filter |
| APF | All-pass filter |
| Peaking | Peaking EQ (parametric boost/cut) |
| Low Shelf | Low-frequency shelving EQ |
| High Shelf | High-frequency shelving EQ |

## File Inventory

| File | Role |
|------|------|
| `rbj_eq.py` | Core Python library: coefficient design, Biquad, EQBand, ParametricEQ, plotting |
| `rbj_eq.hpp` | Core C++ header-only library: same architecture as the Python version |
| `live_eq.py` | Real-time EQ on a Logitech USB headset (mono mic → stereo phones) |
| `live_eq_old.py` | Earlier version of the live EQ with stereo L/R processing |
| `live_eq_pulse.py` | Real-time EQ via PulseAudio with stereo I/O |
| `headset_talkback.py` | EQ experimentation script (pass-through, SSB shaping, notches, BPF for FSK) |
| `plot_eq.py` | Standalone frequency-response plotting of a ParametricEQ |
| `gen_golden.py` | Generates `rbj_golden.txt` reference data from the Python implementation |
| `rbj_golden.txt` | Golden reference: coefficients + 64-sample impulse responses for 9 test cases |
| `test_rbj.cpp` | C++ test: reads `rbj_golden.txt`, compares C++ output to Python reference |
| `CMakeLists.txt` | CMake build for `test_rbj.cpp` |
| `biquadcookbook.txt` | The original RBJ Audio EQ Cookbook text (reference material) |
| `query_devices` | Shell one-liner to list available audio devices via `sounddevice` |
| `setupNULLSink` | Shell script stub for creating a PulseAudio null sink |
| `tags` | ctags index of `rbj_eq.hpp` and `test_rbj.cpp` |

## Architecture

```
biquadcookbook.txt  (reference formulae)
        │
        ▼
┌──────────────┐     ┌──────────────┐
│  rbj_eq.py   │     │ rbj_eq.hpp   │
│  (Python)    │     │ (C++)        │
│              │     │              │
│ design_*()   │     │ design_*()   │
│ Biquad       │     │ Biquad       │
│ EQBand       │     │ EQBand       │
│ ParametricEQ │     │ ParametricEQ │
│ freqz_*()    │     └──────┬───────┘
│ plot_*()     │            │
└──────┬───────┘            │
       │                    │
       ├─── gen_golden.py ──┼─── rbj_golden.txt
       │                    │           │
       │                    │     test_rbj.cpp
       │                    │
       ├─── live_eq.py          (Logitech headset)
       ├─── live_eq_pulse.py    (PulseAudio)
       ├─── headset_talkback.py (experimentation)
       └─── plot_eq.py          (visualization)
```

## Dependencies

- **Python**: `numpy`, `matplotlib`, `sounddevice` (PortAudio binding)
- **C++**: C++17 standard library only (header-only, no external deps)
- **System**: PulseAudio (for `live_eq_pulse.py` and `setupNULLSink`)
