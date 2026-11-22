# Cross-validation against FOMCON

[FOMCON](https://fomcon.net) is the reference MATLAB toolbox for
fractional-order modelling and control. This folder contains the script that
generates reference data with FOMCON and the data themselves, which are
compared against pyfopid by `tests/test_fomcon_reference.py`.

## Cases

| Case | FOMCON | pyfopid | Criterion |
|---|---|---|---|
| 1. Oustaloup approximation of `s^α`, α ∈ {±0.3, ±0.5, −0.9, 0.8}, band `[1e-3, 1e3]`, N = 5 | `oustapp(G, wb, wh, N, 'oust')` | `oustaloup(α, wb, wh, N)` | Identical response for \|α\| ≤ 0.5; for \|α\| > 0.5 pyfopid at least as accurate inside the band |
| 2. FOPID response to `e(t) = sin(2t) + step(t − 1)` | `lsim(fracpid(...), e, t)` | `FOPID.step` (full memory, both integral modes) | Max relative error < 1 % (outside ±0.05 s of the step) |

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
