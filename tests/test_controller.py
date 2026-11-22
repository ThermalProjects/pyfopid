import shutil
import subprocess

import control
import numpy as np
import pytest

from pyfopid import FOPID

MODES = ["accumulated", "gl"]


# ----------------------------------------------------------------- helpers

def classical_pid(errors, kp, ki, kd, dt):
    """Classical discrete PID: backward-rectangle integral, backward difference."""
    out, acc, prev = [], 0.0, 0.0
    for e in errors:
        acc += e * dt
        out.append(kp * e + ki * acc + kd * (e - prev) / dt)
        prev = e
    return np.array(out)


def first_order_loop(controller, tau=1.0, gain=1.0, setpoint=1.0, t_end=40.0):
    """Direct loop: e = r - y, y' = (-y + gain*u)/tau."""
    dt = controller.dt
    y, ys, us = 0.0, [], []
    for _ in range(int(round(t_end / dt))):
        u = controller.step(setpoint - y, measurement=y)
        y += dt * (-y + gain * u) / tau
        ys.append(y)
        us.append(u)
    return np.array(ys), np.array(us)


# ------------------------------------------------------ exact reductions

@pytest.mark.parametrize("mode", MODES)
def test_reduces_to_classical_pid(mode):
    rng = np.random.default_rng(1)
    errors = rng.normal(size=300)
    dt = 0.02
    c = FOPID(kp=1.2, ki=0.7, kd=0.05, lam=1.0, mu=1.0, dt=dt, memory=len(errors), integral=mode)
    u = np.array([c.step(e) for e in errors])
    np.testing.assert_allclose(u, classical_pid(errors, 1.2, 0.7, 0.05, dt), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("mode", MODES)
def test_pure_proportional(mode):
    c = FOPID(kp=3.0, ki=0.0, kd=0.0, lam=0.8, mu=0.4, dt=0.1, memory=10, integral=mode)
    assert [c.step(e) for e in (1.0, -2.0, 0.5)] == pytest.approx([3.0, -6.0, 1.5])


def test_integral_modes_identical_for_full_memory():
    # Exact identity: the partial sums of the GL weights of order 1-lambda are
    # the GL weights of order -lambda, so without truncation the accumulated
    # and the direct GL integral coincide to round-off.
    dt, n = 1e-3, 2000
    t = np.arange(n) * dt
    errors = np.sin(3 * t)
    a = FOPID(kp=0, ki=1, kd=0, lam=0.7, mu=0, dt=dt, memory=n, integral="accumulated")
    b = FOPID(kp=0, ki=1, kd=0, lam=0.7, mu=0, dt=dt, memory=n, integral="gl")
    ua = np.array([a.step(e) for e in errors])
    ub = np.array([b.step(e) for e in errors])
    np.testing.assert_allclose(ua, ub, rtol=1e-9, atol=1e-12)


# ------------------------------------------------------- closed loop

@pytest.mark.parametrize("mode", MODES)
def test_closed_loop_tracks_setpoint(mode):
    c = FOPID(kp=2.0, ki=1.0, kd=0.1, lam=0.9, mu=0.5, dt=0.01, memory=4000, integral=mode)
    y, _ = first_order_loop(c)
    assert abs(y[-1] - 1.0) < 0.02


def test_accumulated_removes_steady_state_error_with_short_memory():
    # With a 0.3 s memory the GL integral has finite DC gain and leaves a
    # residual error; the accumulated scheme keeps true integral action.
    kw = dict(kp=0.5, ki=0.5, kd=0.0, lam=0.8, mu=0.0, dt=0.01, memory=30)
    y_acc, _ = first_order_loop(FOPID(**kw, integral="accumulated"), t_end=200)
    y_gl, _ = first_order_loop(FOPID(**kw, integral="gl"), t_end=200)
    err_acc, err_gl = abs(1 - y_acc[-1]), abs(1 - y_gl[-1])
    assert err_acc < 1e-3
    assert err_gl > 10 * err_acc


@pytest.mark.parametrize("mode", MODES)
def test_saturation_and_anti_windup(mode):
    c = FOPID(kp=5.0, ki=5.0, kd=0.0, lam=0.9, mu=0.0, dt=0.01, memory=4000,
              u_min=0.0, u_max=1.5, integral=mode)
    y, u = first_order_loop(c, setpoint=1.0)
    assert u.max() <= 1.5 and u.min() >= 0.0
    assert abs(y[-1] - 1.0) < 0.02
    assert y.max() - 1.0 < 0.2


def test_integral_limit():
    c = FOPID(kp=0, ki=1, kd=0, lam=1.0, mu=0, dt=0.1, memory=10, integral_limit=0.5)
    for _ in range(100):
        c.step(1.0)
    assert c.integral_state == pytest.approx(0.5)


# ------------------------------------------------------------ derivative

def test_derivative_on_measurement_has_no_setpoint_kick():
    kw = dict(kp=1.0, ki=0.0, kd=1.0, lam=1.0, mu=0.8, dt=0.01, memory=200)
    on_e = FOPID(**kw, derivative_on="error")
    on_y = FOPID(**kw, derivative_on="measurement")
    y = 0.0
    for _ in range(50):  # settle at r = 0
        on_e.step(0.0 - y, measurement=y)
        on_y.step(0.0 - y, measurement=y)
    u_e = on_e.step(1.0 - y, measurement=y)  # set-point step to r = 1
    u_y = on_y.step(1.0 - y, measurement=y)
    assert u_y == pytest.approx(1.0)          # only Kp * e
    assert u_e > 10 * u_y                     # derivative kick


def test_derivative_on_measurement_sign_convention():
    # Reverse loop (e = y - r) with r = 0: D on measurement (error_sign=-1) must
    # equal D on error. (For r != 0 they differ when mu is non-integer, because
    # the GL derivative of a constant is not zero.)
    kw = dict(kp=0.0, ki=0.0, kd=1.0, lam=1.0, mu=0.6, dt=0.05, memory=100)
    r = 0.0
    a = FOPID(**kw, derivative_on="error")
    b = FOPID(**kw, derivative_on="measurement", error_sign=-1)
    ys = 5.0 + np.sin(np.linspace(0, 6, 120))
    a.reset(error=ys[0] - r)
    b.reset(error=ys[0] - r, measurement=ys[0])
    ua = [a.step(y - r) for y in ys]
    ub = [b.step(y - r, measurement=y) for y in ys]
    np.testing.assert_allclose(ua, ub, atol=1e-10)


def test_measurement_mode_requires_measurement():
    c = FOPID(kp=1, ki=1, kd=1, lam=0.5, mu=0.5, dt=0.1, derivative_on="measurement")
    with pytest.raises(ValueError):
        c.step(1.0)


def test_derivative_filter_attenuates_noise():
    rng = np.random.default_rng(3)
    noise = 0.05 * rng.normal(size=2000)
    kw = dict(kp=0.0, ki=0.0, kd=1.0, lam=1.0, mu=1.24, dt=0.1, memory=30)
    raw = FOPID(**kw)
    filt = FOPID(**kw, d_filter_tau=1.0)
    u_raw = np.array([raw.step(e) for e in noise])
    u_filt = np.array([filt.step(e) for e in noise])
    assert np.std(u_filt) < 0.2 * np.std(u_raw)


def test_derivative_filter_preserves_slow_signals():
    # For a ramp the filtered derivative converges to the unfiltered one.
    dt = 0.01
    ramp = np.arange(3000) * dt
    kw = dict(kp=0.0, ki=0.0, kd=1.0, lam=1.0, mu=1.0, dt=dt, memory=50)
    raw, filt = FOPID(**kw), FOPID(**kw, d_filter_tau=0.2)
    u_raw = [raw.step(e) for e in ramp]
    u_filt = [filt.step(e) for e in ramp]
    assert u_filt[-1] == pytest.approx(u_raw[-1], rel=1e-6)


# --------------------------------------------------------------- reset

def test_reset_clears_history():
    c = FOPID(kp=1.0, ki=1.0, kd=1.0, lam=0.5, mu=0.5, dt=0.1, memory=20)
    first = [c.step(1.0) for _ in range(5)]
    c.reset()
    again = [c.step(1.0) for _ in range(5)]
    assert first == pytest.approx(again)


def test_bumpless_reset_keeps_integral_and_avoids_kick():
    kw = dict(kp=1.0, ki=1.0, kd=0.5, lam=0.8, mu=0.7, dt=0.01, memory=300)
    c = FOPID(**kw)
    for _ in range(2000):
        u_before = c.step(0.2)
    i_before = c.integral_state
    c.reset(error=0.2, keep_integral=True)
    u_after = c.step(0.2)
    assert c.integral_state > i_before          # integral preserved and continuing
    assert abs(u_after - u_before) < 0.05 * abs(u_before)


# --------------------------------------------------- firmware equivalence

def firmware_fopid_core(errors_fn, n, Kp, Ki, Kd, lam, mu, Ts, M, umax, plant):
    """Python transcription of the FOPID core of the Peltier firmware (v6).

    Accumulated GL integral s^-lam = s^-1 s^(1-lam), GL derivative, conditional
    anti-windup and integral clamp |I| <= umax / Ki. The firmware's nonlinear
    derivative regularisation sumD / (1 + |sumD|) and the PWM ramp are not
    reproduced (both are outside the linear FOPID law).
    """
    aV = 1.0 - lam
    wV, wD = np.ones(M + 1), np.ones(M + 1)
    for k in range(1, M + 1):
        wV[k] = wV[k - 1] * ((k - 1) - aV) / k
        wD[k] = wD[k - 1] * ((k - 1) - mu) / k
    TsPowV, TsPowD = Ts**aV, Ts**mu
    e_hist = np.zeros(M + 1)
    Iacc, y, out = 0.0, plant["y0"], []
    for k in range(n):
        e = errors_fn(y, k)
        e_hist[1:] = e_hist[:-1]
        e_hist[0] = e
        v = (wV @ e_hist) / TsPowV
        uP, uD = Kp * e, Kd * (wD @ e_hist) / TsPowD
        uTry = uP + Ki * (Iacc + Ts * v) + uD
        sat_high = uTry > umax and v > 0
        sat_low = uTry < 0 and v < 0
        if not sat_high and not sat_low:
            Iacc += Ts * v
            Ilim = umax / Ki
            Iacc = min(max(Iacc, -Ilim), Ilim)
        u = min(max(uP + Ki * Iacc + uD, 0.0), umax)
        out.append(u)
        y = plant["step"](y, u)
    return np.array(out)


def test_matches_peltier_firmware_in_closed_loop():
    # Gains of the firmware; cooling plant: higher PWM lowers the cold-side temperature.
    Kp, Ki, Kd, lam, mu, Ts, M, umax = 49.9933, 2.9958, 1.7547, 0.7809, 1.2402, 0.1, 30, 255.0
    T_amb, tau, g = 22.0, 40.0, 0.12
    r_profile = lambda k: 12.0 if k < 1500 else 18.0         # set-point step up at 150 s
    plant = dict(y0=T_amb, step=lambda y, u: y + Ts * (-(y - T_amb) - g * u) / tau)
    errors_fn = lambda y, k: y - r_profile(k)                 # firmware convention e = y - r
    n = 3000

    u_fw = firmware_fopid_core(errors_fn, n, Kp, Ki, Kd, lam, mu, Ts, M, umax, plant)

    c = FOPID(kp=Kp, ki=Ki, kd=Kd, lam=lam, mu=mu, dt=Ts, memory=M,
              u_min=0.0, u_max=umax, integral="accumulated", integral_limit=umax / Ki)
    y, u_py = plant["y0"], []
    for k in range(n):
        u = c.step(errors_fn(y, k))
        u_py.append(u)
        y = plant["step"](y, u)

    np.testing.assert_allclose(np.array(u_py), u_fw, rtol=1e-9, atol=1e-9)
    assert (u_fw == umax).any() and (u_fw == 0.0).any()  # saturation exercised


# ---------------------------------------------------------- continuous tf

def test_tf_returns_transfer_function():
    c = FOPID(kp=1.0, ki=0.5, kd=0.2, lam=0.8, mu=0.3, dt=0.01)
    tf = c.tf(1e-2, 1e2, 4)
    assert isinstance(tf, control.TransferFunction)
    w = 1.0
    ideal = 1.0 + 0.5 * (1j * w) ** -0.8 + 0.2 * (1j * w) ** 0.3
    assert abs(np.squeeze(tf(1j * w)) - ideal) / abs(ideal) < 0.02


def test_tf_with_filter():
    c = FOPID(kp=1.0, ki=0.5, kd=0.2, lam=0.8, mu=0.3, dt=0.01, d_filter_tau=0.5)
    w = 1.0
    ideal = 1.0 + 0.5 * (1j * w) ** -0.8 + 0.2 * (1j * w) ** 0.3 / (0.5j * w + 1)
    assert abs(np.squeeze(c.tf(1e-2, 1e2, 4)(1j * w)) - ideal) / abs(ideal) < 0.02


# --------------------------------------------------------------- C export

C_CONFIGS = [
    dict(integral="accumulated"),
    dict(integral="accumulated", derivative_on="measurement", error_sign=-1,
         d_filter_tau=0.3, integral_limit=2.0),
    dict(integral="gl"),
    dict(integral="gl", derivative_on="measurement", d_filter_tau=0.1),
]


@pytest.mark.parametrize("cfg", C_CONFIGS)
def test_c_header_compiles_and_matches_python(tmp_path, cfg):
    cc = shutil.which("gcc") or shutil.which("cc") or shutil.which("clang")
    if cc is None:
        pytest.skip("no C compiler available")
    kw = dict(kp=2, ki=1, kd=0.1, lam=0.9, mu=0.5, dt=0.01, memory=16, u_min=0, u_max=1.8)
    kw.update(cfg)
    errors = [1.0, 0.8, 0.5, 0.2, -0.1, -0.3, 0.0, 0.4, 0.9, 1.2] * 3
    meas = [0.1 * k for k in range(len(errors))]
    (tmp_path / "fopid.h").write_text(FOPID(**kw).to_c_header("fopid"))
    ev = ", ".join(f"{e:.6f}f" for e in errors)
    yv = ", ".join(f"{y:.6f}f" for y in meas)
    (tmp_path / "main.c").write_text(
        '#include <stdio.h>\n#include "fopid.h"\n'
        f"static const float E[] = {{{ev}}};\nstatic const float Y[] = {{{yv}}};\n"
        "int main(void){ fopid_state_t s; fopid_reset(&s, 0.0f, 0.0f);\n"
        f"  for (int k = 0; k < {len(errors)}; ++k) printf(\"%.9g\\n\", fopid_step(&s, E[k], Y[k]));\n"
        "  return 0; }\n"
    )
    exe = tmp_path / "main"
    subprocess.run([cc, "-std=c99", "-Wall", "-Wextra", "-Werror", str(tmp_path / "main.c"),
                    "-o", str(exe), "-lm"], check=True, cwd=tmp_path)
    out = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout.split()
    c_vals = np.array([float(v) for v in out])
    c = FOPID(**kw)
    py_vals = np.array([c.step(e, measurement=y) for e, y in zip(errors, meas)])
    np.testing.assert_allclose(c_vals, py_vals, rtol=1e-4, atol=1e-5)


def test_c_header_validation():
    c = FOPID(kp=1.0, ki=0.5, kd=0.2, lam=0.8, mu=0.3, dt=0.01, memory=8)
    assert "#define TEC_N 9" in c.to_c_header("tec")
    with pytest.raises(ValueError):
        c.to_c_header("not valid")


# ------------------------------------------------------------ validation

@pytest.mark.parametrize(
    "kwargs",
    [dict(lam=0.0), dict(mu=-0.1), dict(dt=0.0), dict(memory=0), dict(u_min=1.0, u_max=0.0),
     dict(integral="foo"), dict(lam=2.5), dict(derivative_on="foo"), dict(d_filter_tau=-1),
     dict(error_sign=0), dict(integral_limit=0)],
)
def test_validation(kwargs):
    base = dict(kp=1.0, ki=1.0, kd=1.0, lam=0.5, mu=0.5, dt=0.1)
    base.update(kwargs)
    with pytest.raises(ValueError):
        FOPID(**base)
