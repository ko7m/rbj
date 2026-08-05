# Live Audio Applications

## What These Files Do

Four Python scripts apply the `rbj_eq.py` parametric EQ to live audio in
real-time using the `sounddevice` library (a PortAudio binding). They represent
an evolution of the same idea — pipe audio through biquad EQ bands with
near-zero latency — across different hardware and software configurations.

## `live_eq.py` — Logitech USB Headset

**Purpose**: Real-time EQ on a specific USB headset (device index 0).

- **Input**: 1 channel (headset mic)
- **Output**: 2 channels (stereo headphones)
- **Routing**: Mono mic → EQ → duplicated to both ears
- **Block size**: 256 samples at 48 kHz (~5.3 ms latency)
- **EQ config**: All bands commented out — effectively a pass-through as
  shipped. Commented-out presets include a subtle 3-band EQ and a "massive
  bass boost / kill highs" configuration.

The callback reads `indata[:, 0]`, processes it through a single mono
`ParametricEQ`, and writes the result to both output channels.

## `live_eq_old.py` — Stereo Default Device

**Purpose**: Earlier iteration with stereo L/R processing on the default
audio device.

- **Input**: 2 channels
- **Output**: 2 channels
- **Routing**: Independent L/R EQ chains (cloned band parameters)
- **EQ config**: 3 active bands — low shelf +6 dB at 100 Hz, peaking -3 dB
  at 2 kHz, high shelf +4 dB at 8 kHz

Uses `sd.Stream(channels=2)` with no explicit device selection (default
device). The callback copies `indata`, processes L and R through separate
`ParametricEQ` instances, and writes back.

## `live_eq_pulse.py` — PulseAudio Stereo

**Purpose**: Real-time stereo EQ via PulseAudio named device (`'pulse'`).

- **Input**: 2 channels (PulseAudio source)
- **Output**: 2 channels (PulseAudio sink)
- **Routing**: Independent L/R EQ chains
- **EQ config**: Controlled by `eqActive` flag (default 0 = pass-through).
  When active, two commented-out presets are available:
  - "boomy": low shelf +12 dB, peaking -15 dB at 2 kHz, high shelf -12 dB
  - "thin and bright": high shelf +15 dB at 4 kHz, low shelf -12 dB at 200 Hz

Prints the full device list and default device at startup for diagnostics.
Uses `device=('pulse', 'pulse')` to explicitly select PulseAudio for both
input and output.

## `headset_talkback.py` — Duplicate of `live_eq.py`

**Correction**: earlier revisions of this doc described `headset_talkback.py`
as a PulseAudio experimentation workbench (SSB shaping, carrier notch, FSK
bandpass, etc.). That description was misattributed — it actually belongs to
`plot_eq.py` (see [`04-utilities-and-scripts.md`](04-utilities-and-scripts.md)).

As it stands, `headset_talkback.py` is **byte-for-byte identical** to
`live_eq.py` (same `md5sum`, unchanged since the initial commit `2031ede`) —
the Logitech USB headset pass-through script, not a distinct talkback
experiment. It has the same purpose, input/output shape, and commented-out
presets as `live_eq.py` above.

This looks like a stray copy-paste left over from scaffolding the project
rather than a deliberate second script. If a real "talkback" variant was
intended, it was never written — worth either deleting this file (it adds no
value over `live_eq.py`) or fleshing it out into whatever it was meant to be
(e.g. sending processed audio to a monitor/talkback bus instead of the main
headphone output).

## Common Patterns

All scripts share the same callback structure:

1. Extract per-channel data from `indata`
2. Call `eq.process_block(x)` for each channel
3. Write results to `outdata`
4. Main loop: `sd.sleep(100)` or busy-wait until Ctrl+C

All use 48 kHz sample rate and 256-sample block size.
