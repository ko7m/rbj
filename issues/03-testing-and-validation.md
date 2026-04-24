# Testing and Cross-Language Validation

## What This Does

A golden-file test strategy validates that the C++ port (`rbj_eq.hpp`)
produces identical results to the Python reference implementation
(`rbj_eq.py`). The Python side generates reference data; the C++ side
reads it and compares.

## Components

### `gen_golden.py` — Golden Data Generator

Runs 9 test cases covering all filter types through the Python
implementation and writes `rbj_golden.txt` with:

1. **Filter parameters**: kind, f0, Q, gain_db, S, Fs, impulse length
2. **Normalized coefficients**: b0, b1, b2, a1, a2 (17 significant digits)
3. **64-sample impulse response**: the output of feeding `[1, 0, 0, ..., 0]`
   through a Python `Biquad` initialized with those coefficients

#### Test Cases

| # | Type | f0 (Hz) | Q | Gain (dB) | S |
|---|------|---------|---|-----------|---|
| 0 | LPF | 1000 | 0.707 | 0 | 1 |
| 1 | HPF | 200 | 0.5 | 0 | 1 |
| 2 | BPF | 3000 | 2.0 | 0 | 1 |
| 3 | Notch | 1000 | 1.0 | 0 | 1 |
| 4 | APF | 500 | 0.7 | 0 | 1 |
| 5 | Peaking | 4000 | 1.0 | +6 | 1 |
| 6 | Peaking | 4000 | 1.0 | -6 | 1 |
| 7 | Low Shelf | 200 | 0.707 | +6 | 1 |
| 8 | High Shelf | 8000 | 0.707 | -3 | 1 |

All at Fs = 48000 Hz. Tests 5 and 6 validate that peaking boost and cut
are complementary.

### `rbj_golden.txt` — Reference Data

Plain text file, human-readable. Each test block has:

```
# test N
<kind> <f0> <Q> <gain_db> <S> <Fs> <N_samples>
coeffs <b0> <b1> <b2> <a1> <a2>
impulse <y0> <y1> ... <y63>
```

All values use `%.17g` formatting for full double-precision fidelity.

### `test_rbj.cpp` — C++ Validation

Reads `rbj_golden.txt`, and for each test case:

1. Computes C++ coefficients via `rbj::design_*()` functions
2. Compares against Python reference coefficients (tolerance: 1e-12)
3. Computes a 64-sample impulse response via `rbj::Biquad::process_block()`
4. Compares against Python reference impulse response (tolerance: 1e-10)

Reports PASS/FAIL per test and prints detailed mismatch info on failure
(first 10 samples of impulse, all 5 coefficients).

### `CMakeLists.txt` — Build System

- C++17, single-target build
- Copies `rbj_golden.txt` into the build directory so the test can find it
- Provides a `run_tests` custom target: `cmake --build . --target run_tests`

## How to Run

```bash
# Generate fresh golden data (optional, already committed)
python3 gen_golden.py

# Build and run C++ tests
mkdir -p build && cd build
cmake ..
make run_tests
```

## Tolerance Rationale

- **Coefficient tolerance (1e-12)**: Both implementations use the same
  double-precision formulae; differences arise only from expression
  evaluation order and compiler optimizations.
- **Impulse tolerance (1e-10)**: Accumulated rounding through 64 DF2T
  iterations can amplify per-sample errors, hence the looser bound.
