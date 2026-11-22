"""Cross-validation against reference data generated with FOMCON (MATLAB).

The data are produced by ``validation/fomcon/generate_reference.m`` and stored
in ``validation/fomcon/data``. Tests are skipped if the files are absent.
"""

from pathlib import Path

import numpy as np
import pytest

from pyfopid import FOPID, freqresp_zpk, oustaloup

DATA = Path(__file__).resolve().parents[1] / "validation" / "fomcon" / "data"


def _load(name):
    path = DATA / name
    if not path.exists():
        pytest.skip(f"FOMCON reference file not found: {path.name}")
    arr = np.genfromtxt(path, delimiter=",", names=True)
    return np.atleast_1d(arr)


# --------------------------------------------------------------- case 1

def _case1_groups():
    d = _load("case1_oustaloup.csv")
    for a in np.unique(d["alpha"]):
        yield float(a), d[d["alpha"] == a]


def test_oustaloup_matches_fomcon_for_small_orders():
    """For |alpha| <= 0.5 both implementations approximate the same s^alpha
    with the same formula, so the responses must coincide."""
    checked = 0
    for a, g in _case1_groups():
        if abs(a) > 0.5:
            continue
        wb, wh, N = g["wb"][0], g["wh"][0], int(g["N"][0])
        h = freqresp_zpk(oustaloup(a, wb, wh, N, as_tf=False), g["w"])
        np.testing.assert_allclose(20 * np.log10(np.abs(h)), g["mag_db"], atol=1e-6)
        np.testing.assert_allclose(np.degrees(np.angle(h)), g["phase_deg"], atol=1e-6)
        checked += 1
    assert checked > 0


def test_oustaloup_at_least_as_accurate_as_fomcon_for_large_orders():
    """For |alpha| > 0.5 pyfopid realises the nearest integer exactly and
    approximates only the remainder; FOMCON approximates alpha directly.
    Inside the band pyfopid must not be less accurate than FOMCON."""
    checked = 0
    for a, g in _case1_groups():
        if abs(a) <= 0.5:
            continue
        wb, wh, N = g["wb"][0], g["wh"][0], int(g["N"][0])
        inner = (g["w"] >= 10 * wb) & (g["w"] <= wh / 10)
        w = g["w"][inner]
        ideal_db = 20 * a * np.log10(w)
        ideal_ph = 90.0 * a
        h = freqresp_zpk(oustaloup(a, wb, wh, N, as_tf=False), w)
        err_py = np.max(np.abs(20 * np.log10(np.abs(h)) - ideal_db))
        err_fc = np.max(np.abs(g["mag_db"][inner] - ideal_db))
        ph_py = np.max(np.abs(np.degrees(np.angle(h)) - ideal_ph))
        ph_fc = np.max(np.abs(g["phase_deg"][inner] - ideal_ph))
        assert err_py <= err_fc + 1e-3
        assert ph_py <= ph_fc + 1e-2
        checked += 1
    assert checked > 0


# --------------------------------------------------------------- case 2

def test_fopid_response_matches_fomcon_lsim():
    """Open-loop controller response u = C(s) e for a sine-plus-step error.

    FOMCON simulates the FOTF with its Grünwald-Letnikov based lsim; pyfopid
    runs the discrete controller with full memory. Both evaluate the same GL
    sums, so the responses coincide to round-off (measured: 7.5e-12 relative,
    FOMCON 1.50.4, MATLAB R2026b), including at the step discontinuity.
    With full memory the accumulated and direct integral modes are identical,
    since the partial sums of the GL weights of order 1-lambda are the GL
    weights of order -lambda.
    """
    p = _load("case2_fopid_params.csv")[0]
    d = _load("case2_fopid_response.csv")
    for mode in ("gl", "accumulated"):
        c = FOPID(kp=p["kp"], ki=p["ki"], kd=p["kd"], lam=p["lam"], mu=p["mu"],
                  dt=p["dt"], memory=d.size, integral=mode)
        u = np.array([c.step(e) for e in d["e"]])
        rel_err = np.max(np.abs(u - d["u"])) / np.max(np.abs(d["u"]))
        assert rel_err < 1e-9, f"{mode}: max relative error {rel_err:.2e}"
