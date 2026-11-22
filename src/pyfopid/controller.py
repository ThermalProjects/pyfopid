"""Discrete-time fractional-order PID controller, PI^lambda D^mu.

Continuous law:

    u(t) = Kp e(t) + Ki D^(-lambda) e(t) + Kd D^(mu) e(t)

Integral term
-------------
Two Grünwald-Letnikov (GL) discretisations are provided:

``integral="accumulated"`` (default)
    Uses the factorisation s^(-lambda) = s^(-1) * s^(1 - lambda):

        v_k = dt^(-(1-lambda)) * sum_{j=0}^{L} w_j(1-lambda) e_{k-j}
        I_k = I_{k-1} + dt * v_k

    The integer integrator keeps genuine integral action (infinite DC gain,
    zero steady-state error) even with a short memory L, whereas the
    fractional part is confined to the GL sum.

``integral="gl"``
    Direct GL sum of order -lambda with memory L:

        I_k = dt^lambda * sum_{j=0}^{L} w_j(-lambda) e_{k-j}

    With a finite memory the DC gain is finite, so a small steady-state error
    remains; it is provided for comparison with the literature.

Derivative term
---------------
    D_k = dt^(-mu) * sum_{j=0}^{L} w_j(mu) d_{k-j}

where d is the error (``derivative_on="error"``) or the sign-adjusted
measurement (``derivative_on="measurement"``, no set-point kick). An optional
first-order low-pass filter with time constant ``d_filter_tau``,
1 / (tau s + 1), discretised with backward Euler, limits noise amplification
(important for mu > 1).

With lambda = mu = 1, no filter and derivative on error, both integral modes
reduce exactly to the classical discrete PID (backward-rectangle integral,
backward-difference derivative).
"""

from __future__ import annotations

import numpy as np

from .approx import oustaloup
from .operators import gl_weights

__all__ = ["FOPID"]

_INTEGRAL_MODES = ("accumulated", "gl")
_DERIVATIVE_MODES = ("error", "measurement")


class FOPID:
    """Fractional-order PID controller for sampled-data loops.

    Parameters
    ----------
    kp, ki, kd : float
        Proportional, integral and derivative gains.
    lam : float
        Integration order lambda (> 0). For ``integral="accumulated"`` it must
        satisfy 0 < lam <= 2.
    mu : float
        Differentiation order mu (>= 0).
    dt : float
        Sampling period in seconds.
    memory : int
        Short-memory length L (number of past samples in the GL sums). The
        fractional behaviour of the integral is effective for frequencies above
        roughly 1 / (L * dt) rad/s; choose L so that L * dt covers the dominant
        time constants of the plant.
    u_min, u_max : float, optional
        Output saturation limits. Conditional-integration anti-windup is
        applied: a sample is not integrated if the output is saturated and
        integrating it would push further into saturation.
    integral : {"accumulated", "gl"}
        Discretisation of the integral term (see module docstring).
    integral_limit : float, optional
        Symmetric bound on the integral state ``|I| <= integral_limit``
        (``accumulated`` mode only). Extra protection on top of anti-windup.
    derivative_on : {"error", "measurement"}
        Signal differentiated by the D term.
    d_filter_tau : float, optional
        Time constant (s) of the first-order derivative filter. ``None`` or 0
        disables it.
    error_sign : {+1, -1}
        Error convention, needed only for ``derivative_on="measurement"``:
        +1 for e = r - y (heating / direct loops), -1 for e = y - r (e.g.
        cooling with a Peltier module where a larger u lowers y).

    Examples
    --------
    >>> c = FOPID(kp=1.0, ki=0.5, kd=0.1, lam=0.9, mu=0.6, dt=0.01, memory=500)
    >>> u = c.step(1.0)
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        lam: float,
        mu: float,
        dt: float,
        memory: int = 1000,
        u_min: float | None = None,
        u_max: float | None = None,
        integral: str = "accumulated",
        integral_limit: float | None = None,
        derivative_on: str = "error",
        d_filter_tau: float | None = None,
        error_sign: int = 1,
    ) -> None:
        if lam <= 0:
            raise ValueError("lam must be > 0")
        if integral not in _INTEGRAL_MODES:
            raise ValueError(f"integral must be one of {_INTEGRAL_MODES}")
        if integral == "accumulated" and lam > 2:
            raise ValueError("integral='accumulated' requires 0 < lam <= 2")
        if mu < 0:
            raise ValueError("mu must be >= 0")
        if dt <= 0:
            raise ValueError("dt must be > 0")
        if memory < 1:
            raise ValueError("memory must be >= 1")
        if u_min is not None and u_max is not None and u_min >= u_max:
            raise ValueError("u_min must be < u_max")
        if derivative_on not in _DERIVATIVE_MODES:
            raise ValueError(f"derivative_on must be one of {_DERIVATIVE_MODES}")
        if d_filter_tau is not None and d_filter_tau < 0:
            raise ValueError("d_filter_tau must be >= 0")
        if error_sign not in (1, -1):
            raise ValueError("error_sign must be +1 or -1")
        if integral_limit is not None and integral_limit <= 0:
            raise ValueError("integral_limit must be > 0")

        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.lam, self.mu, self.dt = float(lam), float(mu), float(dt)
        self.memory = int(memory)
        self.u_min, self.u_max = u_min, u_max
        self.integral = integral
        self.integral_limit = integral_limit
        self.derivative_on = derivative_on
        self.d_filter_tau = float(d_filter_tau) if d_filter_tau else 0.0
        self.error_sign = int(error_sign)

        n = self.memory + 1
        if integral == "accumulated":
            a = 1.0 - self.lam
            # dI_k = cv . e_hist  (already includes dt and dt^-(1-lambda))
            self._cv = self.dt * self.dt ** (-a) * gl_weights(a, n)
        else:
            # I_k = ci . e_hist   (gain Ki folded in)
            self._ci = self.ki * self.dt**self.lam * gl_weights(-self.lam, n)
        self._cd = self.kd * self.dt ** (-self.mu) * gl_weights(self.mu, n)
        self._d_alpha = self.dt / (self.d_filter_tau + self.dt)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(
        self,
        error: float = 0.0,
        measurement: float | None = None,
        keep_integral: bool = False,
    ) -> None:
        """Reset the controller histories.

        Parameters
        ----------
        error : float
            Value used to fill the error history. Passing the current error
            (instead of 0) avoids an artificial step at the first sample.
        measurement : float, optional
            Value used to fill the measurement history when
            ``derivative_on="measurement"``. Defaults to 0.
        keep_integral : bool
            Keep the integral state (``accumulated`` mode). Use it, together
            with ``error``, for bumpless set-point changes.
        """
        n = self.memory + 1
        self._hist_e = np.full(n, float(error))
        if self.derivative_on == "error":
            self._hist_d = np.full(n, float(error))
        else:
            y0 = 0.0 if measurement is None else float(measurement)
            self._hist_d = np.full(n, -self.error_sign * y0)
        if not keep_integral or not hasattr(self, "_I"):
            self._I = 0.0
        self._d_filt = 0.0
        self._d_initialised = False

    @property
    def integral_state(self) -> float:
        """Current integral state I (accumulated mode) or GL integral value."""
        if self.integral == "accumulated":
            return self._I
        return float(self._ci @ self._hist_e) / self.ki if self.ki else 0.0

    # --------------------------------------------------------------- helpers
    def _saturate(self, u: float) -> float:
        if self.u_max is not None and u > self.u_max:
            return self.u_max
        if self.u_min is not None and u < self.u_min:
            return self.u_min
        return u

    @staticmethod
    def _push(hist: np.ndarray, value: float) -> None:
        hist[1:] = hist[:-1]
        hist[0] = value

    def _derivative(self) -> float:
        d_raw = float(self._cd @ self._hist_d)
        if self.d_filter_tau == 0.0:
            return d_raw
        if not self._d_initialised:
            self._d_filt = d_raw
            self._d_initialised = True
        else:
            self._d_filt += self._d_alpha * (d_raw - self._d_filt)
        return self._d_filt

    # ------------------------------------------------------------------ step
    def step(self, error: float, measurement: float | None = None) -> float:
        """Advance one sample and return the (saturated) control action.

        Parameters
        ----------
        error : float
            Control error at this sample (sign convention: see ``error_sign``).
        measurement : float, optional
            Process measurement; required when ``derivative_on="measurement"``.
        """
        e = float(error)
        self._push(self._hist_e, e)
        if self.derivative_on == "error":
            self._push(self._hist_d, e)
        else:
            if measurement is None:
                raise ValueError("derivative_on='measurement' requires `measurement`")
            self._push(self._hist_d, -self.error_sign * float(measurement))

        u_pd = self.kp * e + self._derivative()

        if self.integral == "accumulated":
            d_i = float(self._cv @ self._hist_e)
            u_try = u_pd + self.ki * (self._I + d_i)
            u_sat = self._saturate(u_try)
            winding = u_sat != u_try and self.ki * d_i * (u_try - u_sat) > 0
            if not winding:
                self._I += d_i
                if self.integral_limit is not None:
                    self._I = min(max(self._I, -self.integral_limit), self.integral_limit)
            return self._saturate(u_pd + self.ki * self._I)

        # integral == "gl"
        u = u_pd + float(self._ci @ self._hist_e)
        u_sat = self._saturate(u)
        if u_sat != u and e * (u - u_sat) > 0:
            self._hist_e[0] = 0.0
            u_sat = self._saturate(u_pd + float(self._ci @ self._hist_e))
        return u_sat

    # ------------------------------------------------------------ continuous
    def tf(self, wb: float = 1e-3, wh: float = 1e3, N: int = 5):
        """Continuous transfer function using Oustaloup approximations.

        Returns ``Kp + Ki * s^(-lam) + Kd * s^(mu) / (tau s + 1)`` as a
        :class:`control.TransferFunction` (the filter factor only if
        ``d_filter_tau > 0``). The integer part of each order is realised
        exactly. If ``mu >= 0.5`` and no filter is used, the result is
        improper, as is the ideal derivative.
        """
        import control

        c = control.tf([self.kp], [1.0])
        if self.ki != 0.0:
            c = c + self.ki * oustaloup(-self.lam, wb, wh, N)
        if self.kd != 0.0:
            d = self.kd * oustaloup(self.mu, wb, wh, N)
            if self.d_filter_tau > 0.0:
                d = d * control.tf([1.0], [self.d_filter_tau, 1.0])
            c = c + d
        return c

    # ------------------------------------------------------------- C export
    def to_c_header(self, name: str = "fopid", precision: int = 9) -> str:
        """Generate a self-contained C99 header implementing this controller.

        Weights are pre-scaled by the gains and sampling period. The generated
        step function is ``float <name>_step(<name>_state_t *s, float e, float y)``;
        ``y`` (measurement) is used only when ``derivative_on="measurement"``.
        On small AVR boards keep the memory short or move the arrays to flash
        with PROGMEM.
        """
        if not name.isidentifier():
            raise ValueError("name must be a valid C identifier")
        up = name.upper()
        n = self.memory + 1

        def cf(v: float) -> str:
            s = f"{float(v):.{precision}g}"
            if not any(ch in s for ch in ".e"):
                s += ".0"
            return s + "f"

        def arr(a: np.ndarray) -> str:
            return "{" + ", ".join(cf(v) for v in a) + "}"

        umin = "-INFINITY" if self.u_min is None else cf(self.u_min)
        umax = "INFINITY" if self.u_max is None else cf(self.u_max)
        acc = self.integral == "accumulated"
        ilim = "INFINITY" if self.integral_limit is None else cf(self.integral_limit)
        d_meas = self.derivative_on == "measurement"
        filt = self.d_filter_tau > 0.0

        int_array = (
            f"static const float {up}_CV[{up}_N] = {arr(self._cv)};"
            if acc
            else f"static const float {up}_CI[{up}_N] = {arr(self._ci)};"
        )
        d_input = f"{-self.error_sign}.0f * y" if d_meas else "e"
        d_fill = f"{-self.error_sign}.0f * y0" if d_meas else "e0"

        filt_code = (
            f"""    if (!s->d_init) {{ s->d_filt = d; s->d_init = 1; }}
    else s->d_filt += {up}_DALPHA * (d - s->d_filt);
    d = s->d_filt;
"""
            if filt
            else ""
        )

        if acc:
            int_code = f"""    float di = {name}_dot({up}_CV, s->hist_e, s->head);
    float u_try = u_pd + {up}_KI * (s->integ + di);
    float u_sat = {name}_clamp(u_try);
    if (!(u_sat != u_try && {up}_KI * di * (u_try - u_sat) > 0.0f)) {{
        s->integ += di;
        if (s->integ > {up}_ILIM) s->integ = {up}_ILIM;
        if (s->integ < -{up}_ILIM) s->integ = -{up}_ILIM;
    }}
    return {name}_clamp(u_pd + {up}_KI * s->integ);
"""
        else:
            int_code = f"""    float u = u_pd + {name}_dot({up}_CI, s->hist_e, s->head);
    float u_sat = {name}_clamp(u);
    if (u_sat != u && e * (u - u_sat) > 0.0f) {{
        s->hist_e[s->head] = 0.0f;
        u_sat = {name}_clamp(u_pd + {name}_dot({up}_CI, s->hist_e, s->head));
    }}
    return u_sat;
"""

        return f"""/* Auto-generated by pyfopid. Do not edit by hand.
 * FOPID: Kp={self.kp:g}, Ki={self.ki:g}, Kd={self.kd:g},
 *        lambda={self.lam:g}, mu={self.mu:g}, dt={self.dt:g} s, L={self.memory}
 *        integral={self.integral}, derivative_on={self.derivative_on},
 *        d_filter_tau={self.d_filter_tau:g} s, error_sign={self.error_sign:+d}
 * Usage: call {name}_reset(&s, e0, y0) once, then u = {name}_step(&s, e, y)
 * every dt seconds. y is ignored unless derivative_on = measurement.
 */
#ifndef {up}_H
#define {up}_H
#include <math.h>

#define {up}_N {n}
static const float {up}_KP = {cf(self.kp)};
static const float {up}_KI = {cf(self.ki)};
static const float {up}_UMIN = {umin};
static const float {up}_UMAX = {umax};
static const float {up}_ILIM = {ilim};
static const float {up}_DALPHA = {cf(self._d_alpha)};
{int_array}
static const float {up}_CD[{up}_N] = {arr(self._cd)};

typedef struct {{
    float hist_e[{up}_N];
    float hist_d[{up}_N];
    int head;
    float integ;
    float d_filt;
    int d_init;
}} {name}_state_t;

/* e0, y0: current error and measurement, for a bumpless start. */
static void {name}_reset({name}_state_t *s, float e0, float y0) {{
    (void)y0;
    for (int j = 0; j < {up}_N; ++j) {{ s->hist_e[j] = e0; s->hist_d[j] = {d_fill}; }}
    s->head = 0;
    s->integ = 0.0f;
    s->d_filt = 0.0f;
    s->d_init = 0;
}}

static float {name}_dot(const float *c, const float *h, int head) {{
    float acc = 0.0f;
    int idx = head;
    for (int j = 0; j < {up}_N; ++j) {{
        acc += c[j] * h[idx];
        idx = (idx == 0) ? {up}_N - 1 : idx - 1;
    }}
    return acc;
}}

static float {name}_clamp(float u) {{
    if (u > {up}_UMAX) return {up}_UMAX;
    if (u < {up}_UMIN) return {up}_UMIN;
    return u;
}}

static float {name}_step({name}_state_t *s, float e, float y) {{
    (void)y;
    s->head = (s->head + 1) % {up}_N;
    s->hist_e[s->head] = e;
    s->hist_d[s->head] = {d_input};
    float d = {name}_dot({up}_CD, s->hist_d, s->head);
{filt_code}    float u_pd = {up}_KP * e + d;
{int_code}}}

#endif /* {up}_H */
"""

    def __repr__(self) -> str:
        return (
            f"FOPID(kp={self.kp:g}, ki={self.ki:g}, kd={self.kd:g}, "
            f"lam={self.lam:g}, mu={self.mu:g}, dt={self.dt:g}, memory={self.memory}, "
            f"integral={self.integral!r}, derivative_on={self.derivative_on!r}, "
            f"d_filter_tau={self.d_filter_tau:g})"
        )
