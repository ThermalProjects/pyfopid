"""Closed-loop simulation: FOPID vs classical PID on a first-order-plus-dead-time plant.

Plant: G(s) = K e^{-theta s} / (tau s + 1), simulated with forward Euler.
Run:  python examples/closed_loop.py
"""

import numpy as np

import pyfopid as pf

K, TAU, THETA = 1.0, 2.0, 0.3
DT, T_END = 0.01, 30.0


def simulate(controller):
    n = int(T_END / DT)
    delay = int(THETA / DT)
    y = np.zeros(n)
    u_buf = np.zeros(n)
    for k in range(n - 1):
        u_buf[k] = controller.step(1.0 - y[k])
        u_delayed = u_buf[k - delay] if k >= delay else 0.0
        y[k + 1] = y[k] + DT * (-y[k] + K * u_delayed) / TAU
    return np.arange(n) * DT, y, u_buf


def metrics(t, y):
    overshoot = max(0.0, y.max() - 1.0) * 100
    outside = np.where(np.abs(y - 1.0) > 0.02)[0]
    settling = t[outside[-1] + 1] if outside.size and outside[-1] + 1 < t.size else 0.0
    iae = np.sum(np.abs(1.0 - y)) * DT
    return overshoot, settling, iae


if __name__ == "__main__":
    pid = pf.FOPID(kp=2.0, ki=1.0, kd=0.3, lam=1.0, mu=1.0, dt=DT, memory=3000, u_min=0, u_max=5)
    fopid = pf.FOPID(kp=2.0, ki=1.0, kd=0.3, lam=0.95, mu=0.7, dt=DT, memory=3000, u_min=0, u_max=5,
                      d_filter_tau=0.02)

    print(f"{'controller':<8} {'overshoot %':>12} {'t_s (2%) s':>12} {'IAE':>8}")
    results = {}
    for name, c in (("PID", pid), ("FOPID", fopid)):
        t, y, u = simulate(c)
        results[name] = (t, y)
        os_, ts, iae = metrics(t, y)
        print(f"{name:<8} {os_:12.2f} {ts:12.2f} {iae:8.3f}")

    try:
        import matplotlib.pyplot as plt

        for name, (t, y) in results.items():
            plt.plot(t, y, label=name)
        plt.axhline(1.0, color="k", lw=0.5)
        plt.xlabel("Time (s)")
        plt.ylabel("Output")
        plt.legend()
        plt.tight_layout()
        plt.show()
    except ImportError:
        pass
