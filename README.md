# pyfopid

**Fractional-order modelling and control in Python.**

`pyfopid` provides tested, documented building blocks for fractional-order
control: Grünwald-Letnikov operators, Oustaloup rational approximations,
and a discrete-time FOPID (PI^λD^μ) controller that interoperates with
[python-control](https://python-control.readthedocs.io) and exports to C for
embedded targets (Arduino, ESP32, STM32).

> Status: alpha (0.1.0.dev0). API may change before 1.0.

## Installation

```bash
pip install pyfopid            # once released on PyPI
pip install -e ".[test]"       # development install from a clone
```

Requires Python ≥ 3.10, NumPy, SciPy and python-control ≥ 0.10.

## Quick start

```python
import numpy as np
import pyfopid as pf

# Fractional derivative of order 0.5 of t^2 (Grünwald-Letnikov)
h = 1e-3
t = np.arange(0, 1 + h, h)
d = pf.gl_derivative(t**2, 0.5, h)

# Oustaloup approximation of s^0.5 in [0.01, 100] rad/s as a control.TransferFunction
G = pf.oustaloup(0.5, wb=1e-2, wh=1e2, N=5)

# Discrete FOPID controller (accumulated fractional integral, filtered
# derivative on the measurement, saturation with anti-windup)
c = pf.FOPID(kp=2.0, ki=1.0, kd=0.1, lam=0.9, mu=0.5, dt=0.01, memory=2000,
             u_min=0.0, u_max=12.0,
             derivative_on="measurement", d_filter_tau=0.05)
u = c.step(error=0.3, measurement=24.7)

# Export the same controller to a C header for a microcontroller
open("fopid.h", "w").write(c.to_c_header("fopid"))
```

See [`examples/closed_loop.py`](examples/closed_loop.py) for a full
closed-loop simulation.

## Features (v0.1)

| Module | Contents |
|---|---|
| `operators` | `gl_weights`, `gl_derivative` (with short-memory principle), `fractional_integral`, `caputo_derivative` |
| `approx` | `oustaloup` (any real order; integer part realised exactly), `ZPK`, `freqresp_zpk` |
| `controller` | `FOPID`: accumulated (default) or direct GL fractional integral, derivative on error or measurement, first-order derivative filter, conditional-integration anti-windup and integral clamp, bumpless `reset()`, `.tf()` (Oustaloup-based continuous model), `.to_c_header()` (C99, same options) |

### Design notes

- **Accumulated integral (default).** `s^-λ = s^-1 · s^(1-λ)`: the fractional
  operator of order `1-λ` is evaluated with a GL sum and followed by an
  integer integrator. This keeps genuine integral action (zero steady-state
  error) for any memory length. The direct GL integral (`integral="gl"`) has a
  finite DC gain when the memory is truncated.
- **Memory length.** The fractional behaviour is effective for frequencies
  above roughly `1/(L·dt)` rad/s. Choose `L·dt` comparable to the dominant
  time constants of the plant.
- **Derivative.** For `μ > 1` noise amplification is severe; use
  `d_filter_tau`. With non-integer `μ` the GL derivative of a constant is not
  zero, so differentiating the error and differentiating the measurement are
  not equivalent even for a constant set point; `derivative_on="measurement"`
  also removes the set-point kick.

## Verification

The test suite (`pytest`) checks:

- GL weights against the signed binomial coefficients `(-1)^j C(α, j)`,
  including the sign pattern.
- GL derivatives and integrals against closed-form results
  (`D^α t^k = Γ(k+1)/Γ(k+1−α) t^(k−α)`, `I^α 1 = t^α/Γ(1+α)`) and the
  expected first-order convergence O(h).
- Caputo derivatives (constants map to zero; orders above one with initial
  conditions).
- Oustaloup approximations against `(jω)^α`: with the band
  `[wb, wh]` chosen at least two decades wider than the frequency range of
  interest, errors are < 0.05 dB and < 0.5°. One decade from the band edges
  the phase error rises to ≈ 3° for |α| = 0.5; this is dominated by band
  truncation, not by `N`.
- Exact reduction of `FOPID` to the classical discrete PID for λ = μ = 1 in
  both integral modes.
- Closed-loop tracking with and without saturation; zero steady-state error of
  the accumulated integral with a short memory.
- Sample-by-sample agreement (rtol 1e-9) in closed loop with an independent
  transcription of an FOPID firmware used on a Peltier test bench.
- Compiled C output against Python for all integral/derivative options.

## Roadmap

- [ ] Cross-validation against FOMCON (MATLAB) on reference problems
- [ ] Refined Oustaloup and discrete-time approximations (Tustin/Al-Alaoui CFE)
- [ ] Fractional-order transfer functions (FOTF) with time and frequency responses
- [ ] FOPID tuning (frequency-domain specifications, multi-objective with pymoo)
- [ ] Identification of fractional-order models from data
- [ ] Documentation site and JOSS submission

## Citing

If you use `pyfopid` in research, please cite it using the metadata in
[`CITATION.cff`](CITATION.cff). A DOI will be issued through Zenodo with the
first release.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
