# Changelog

## 0.1.0 (unreleased)

- Grünwald-Letnikov weights, derivatives and integrals with short-memory option.
- Caputo derivative for arbitrary positive order with initial conditions.
- Oustaloup approximation of `s^alpha` for any real order.
- Discrete-time `FOPID` controller:
  - accumulated fractional integral `s^-λ = s^-1 · s^(1-λ)` (default) or direct GL;
  - derivative on error or on measurement, optional first-order filter;
  - conditional-integration anti-windup, optional integral clamp;
  - bumpless `reset(error, measurement, keep_integral)`;
  - continuous `tf()` model and C99 header export with the same options.
