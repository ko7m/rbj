# Utilities and Supporting Files

## `plot_eq.py` — EQ Design Workbench / FSK Filter Visualization

**Correction**: earlier revisions of this doc described `plot_eq.py` as a
simple two-band (low shelf + high shelf) visualization demo. That description
was misattributed — it actually belongs to `headset_talkback.py`, which turns
out to be a byte-for-byte duplicate of `live_eq.py` rather than a distinct
script (see [`02-live-audio-applications.md`](02-live-audio-applications.md)).

The real `plot_eq.py` is a workbench for trying various EQ configurations and
plotting their response via `plot_eq_response()` before committing them to a
live script. Imports `ParametricEQ` and `plot_eq_response` from `rbj_eq.py`
(`blocksize` and `channels` are declared but unused — `plot_eq.py` never
opens an audio stream). Contains many commented-out presets, suggesting it
was used interactively to audition filter shapes:

| Preset | Description |
|--------|-------------|
| Pass-through | No bands (current default `eqL`) |
| SSB-ish shaping | High shelf +6 dB at 300 Hz (S=0.7), low shelf -12 dB at 2.5 kHz (S=0.7) |
| Carrier notch | Notch at 1200 Hz, Q=10 |
| General 3-band | Low shelf +6 dB @ 100 Hz, peaking -3 dB @ 2 kHz, high shelf +4 dB @ 8 kHz |
| Boomy | Low shelf +12 dB @ 200 Hz, peaking -15 dB @ 2 kHz (Q=0.5), high shelf -12 dB @ 4 kHz |
| Thin and bright | High shelf +15 dB @ 4 kHz, low shelf -12 dB @ 200 Hz |
| Stacked notches | 3 notch filters at 1000, 1500, 700 Hz (Q 10/10/20) |
| 1 kHz notch | Narrow notch at 1 kHz (Q=10, BW ~100 Hz) |
| **850 Hz BPF** | **Active**: BPF centered at 850 Hz, BW=250 Hz (Q≈3.4) |

The final active configuration is a 250 Hz wide bandpass filter centered at
850 Hz — this looks like it was being used to isolate an FSK signal
(170 Hz shift keying, mark/space around 850 Hz is a classic RTTY setup).

The script calls `plot_eq_response(eqL, Fs=Fs, worN=4096, show_bands=True,
xlim=(500, 1300), ylim=(-40, 5))` at the end to visualize the active filter
before it would be wired into a live script.

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

## `FilterTypes.odt` — Filter Type Reference Notes

An OpenDocument file added in commit `4f293a6` ("Notes on filter types").
Binary format — contains documentation/notes on the various biquad filter
types and their characteristics. This is a companion to `biquadcookbook.txt`,
likely with additional formatting, diagrams, or personal annotations.

## `issues/agent` — Agent Resume Script

A 3-line bash script that resumes a previous Cursor agent session by ID.
Used for continuing documentation work across sessions.

## `.gitignore`

Standard Python `.gitignore` (generated template). Also ignores `tags`.
