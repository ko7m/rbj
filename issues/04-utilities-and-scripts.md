# Utilities and Supporting Files

## `plot_eq.py` — Standalone EQ Visualization

Imports `ParametricEQ` and `plot_eq_response` from `rbj_eq.py` and renders
a frequency-response plot for a configured EQ. This file is essentially a
scratchpad for visualizing filter curves without running live audio.

Currently builds an EQ with two bands:
- Low shelf: +6 dB at 100 Hz, S=1.0
- High shelf: +4 dB at 8 kHz, S=0.7

Then calls `plot_eq_response()` with `show_bands=True` to display both the
per-band and combined magnitude response on a semilog frequency axis
(20 Hz to Nyquist).

## `query_devices` — Audio Device Listing

A 3-line Python script (invoked as a shell script) that prints all available
audio devices and the current default device using `sounddevice.query_devices()`.

Useful for finding device indices and names to use in the `live_eq*.py`
scripts. Example output includes device name, channel counts, and default
sample rates.

## `setupNULLSink` — PulseAudio Null Sink Setup

A bash script with the actual `pactl load-module module-null-sink` command
commented out. In its current state it just runs `pactl list short sinks` to
show existing PulseAudio sinks.

When the commented line is enabled, it creates a virtual PulseAudio sink
named `eq` (described as "EQ_Sink") that can be used as a loopback target:
route application audio into the null sink, read from its monitor source in
`live_eq_pulse.py`, process it, and send the result to the real hardware
output.

## `biquadcookbook.txt` — Reference Material

The full text of Robert Bristow-Johnson's "Cookbook formulae for audio EQ
biquad filter coefficients." This is the theoretical foundation for the
entire project.

Contains:
- Transfer function definition (Eq 1–4)
- User-defined parameter descriptions (Fs, f0, dBgain, Q, BW, S)
- Intermediate variable derivations (A, w0, alpha)
- All 8 filter coefficient formulae with analog prototypes
- Bilinear transform substitution derivations

## `tags` — ctags Index

An Exuberant Ctags index of `rbj_eq.hpp` and `test_rbj.cpp`. Provides
jump-to-definition data for editors. Lists all classes, structs, member
variables, and functions in the C++ codebase.

## `.gitignore`

Standard Python `.gitignore` (generated template). Also ignores `tags`.
