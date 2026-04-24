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

## `headset_talkback.py` — EQ Experimentation / Talkback

**Purpose**: A workbench for trying various EQ configurations with
PulseAudio stereo I/O. Contains many commented-out presets, suggesting it
was used interactively for experimentation.

- **Input**: 2 channels
- **Output**: 2 channels
- **Routing**: Separate L/R EQ, only L is configured (R clones L's bands)

### Commented-out EQ Presets

| Preset | Description |
|--------|-------------|
| Pass-through | No bands (current default) |
| SSB-ish shaping | High shelf +6 dB at 300 Hz, low shelf -12 dB at 2.5 kHz |
| Carrier notch | Notch at 1200 Hz, Q=10 |
| General 3-band | Low shelf, peaking, high shelf |
| Boomy | Low shelf +12 dB, mid cut, high shelf -12 dB |
| Thin and bright | High shelf +15 dB, low shelf -12 dB |
| Stacked notches | 3 notch filters at 700, 1000, 1500 Hz |
| 1 kHz notch | Narrow notch at 1 kHz (Q=10, BW ~100 Hz) |
| **850 Hz BPF** | **Active**: BPF centered at 850 Hz, BW=250 Hz (Q≈3.4) |

The final active configuration is a 250 Hz wide bandpass filter centered at
850 Hz — this looks like it was being used to isolate an FSK signal
(170 Hz shift keying, mark/space around 850 Hz is a classic RTTY setup).

The script also calls `plot_eq_response()` at the end to visualize the
active filter before streaming (xlim 500–1300 Hz, ylim -40 to +5 dB).

## Common Patterns

All scripts share the same callback structure:

1. Extract per-channel data from `indata`
2. Call `eq.process_block(x)` for each channel
3. Write results to `outdata`
4. Main loop: `sd.sleep(100)` or busy-wait until Ctrl+C

All use 48 kHz sample rate and 256-sample block size.
