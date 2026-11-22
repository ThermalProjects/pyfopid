# Cross-validation against FOMCON

[FOMCON](https://fomcon.net) is the reference MATLAB toolbox for
fractional-order modelling and control. This folder contains the script that
generates reference data with FOMCON and the data themselves, which are
compared against pyfopid by `tests/test_fomcon_reference.py`.

## Cases

| Case | FOMCON | pyfopid | Criterion |
|---|---|---|---|
| 1. Oustaloup approximation of `s^α`, α ∈ {±0.3, ±0.5, −0.9, 0.8}, band `[1e-3, 1e3]`, N = 5 | `oustapp(G, wb, wh, N, 'oust')` | `oustaloup(α, wb, wh, N)` | Identical response for \|α\| ≤ 0.5; for \|α\| > 0.5 pyfopid at least as accurate inside the band |
| 2. FOPID response to `e(t) = sin(2t) + step(t − 1)`, Kp = 1.2, Ki = 0.8, λ = 0.7, Kd = 0.3, μ = 0.6, dt = 1 ms | `lsim(fracpid(...), e, t)` | `FOPID.step` (full memory, both integral modes) | Max relative difference < 1e-9 |

## Results (FOMCON 1.50.4, MATLAB R2026b Update 1, Control System Toolbox 26.2)

**Case 1.** For |α| ≤ 0.5 the two implementations coincide to round-off
(max. difference 5.3e-11 dB and 9.0e-11°). For |α| > 0.5 they differ by
design: FOMCON approximates `s^α` directly, whereas pyfopid realises the
nearest integer order exactly and approximates only the remainder. Maximum
error against the ideal `(jω)^α` in the inner band `[1e-2, 1e2]` rad/s:

| α | pyfopid | FOMCON |
|---|---|---|
| −0.9 | 0.005 dB, 0.54° | 0.039 dB, 5.09° |
| 0.8 | 0.010 dB, 1.09° | 0.035 dB, 4.48° |
| ±0.5 | 0.024 dB, 2.75° | 0.024 dB, 2.75° |
| ±0.3 | 0.015 dB, 1.63° | 0.015 dB, 1.63° |

**Case 2.** Maximum relative difference over 5001 samples, including the step
discontinuity: 7.5e-12 (RMS 2.0e-12) for both integral modes. With full
memory the accumulated and direct GL integrals are mathematically identical
(the partial sums of the GL weights of order 1−λ are the GL weights of
order −λ); they differ only when the memory is truncated.

## Regenerate

```matlab
>> cd validation/fomcon
>> generate_reference
```

Then, from the repository root:

```
pytest tests/test_fomcon_reference.py -v
```

The MATLAB and toolbox versions used are recorded in `data/environment.txt`.
