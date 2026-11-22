import control
import numpy as np
import pytest

from pyfopid import ZPK, freqresp_zpk, oustaloup


def _band_errors(alpha, wb, wh, N, margin_decades):
    zpk = oustaloup(alpha, wb, wh, N, as_tf=False)
    w = np.logspace(np.log10(wb) + margin_decades, np.log10(wh) - margin_decades, 400)
    h = freqresp_zpk(zpk, w)
    ideal = (1j * w) ** alpha
    mag_err_db = np.abs(20 * np.log10(np.abs(h) / np.abs(ideal))).max()
    phase_err_deg = np.abs(np.degrees(np.angle(h / ideal))).max()
    return mag_err_db, phase_err_deg


@pytest.mark.parametrize("alpha", [-0.9, -0.5, 0.3, 0.5, 0.8, 1.3, -1.6])
def test_oustaloup_accuracy_two_decades_inside_band(alpha):
    # Recommended usage: band at least two decades wider than the range of interest.
    mag, phase = _band_errors(alpha, 1e-3, 1e3, 6, margin_decades=2)
    assert mag < 0.05
    assert phase < 0.5


@pytest.mark.parametrize("alpha", [-0.5, 0.5, -1.6])
def test_oustaloup_band_edge_degradation_is_bounded(alpha):
    # One decade inside the band the phase error grows (known behaviour of the
    # method, dominated by band truncation rather than by N), but stays bounded.
    mag, phase = _band_errors(alpha, 1e-2, 1e2, 6, margin_decades=1)
    assert mag < 0.1
    assert phase < 3.0


def test_oustaloup_integer_order_is_exact():
    zpk = oustaloup(1.0, as_tf=False)
    np.testing.assert_array_equal(zpk.zeros, [0.0])
    assert zpk.poles.size == 0 and zpk.gain == 1.0
    zpk = oustaloup(-2.0, as_tf=False)
    np.testing.assert_array_equal(zpk.poles, [0.0, 0.0])


def test_oustaloup_structure():
    zpk = oustaloup(0.5, 1e-3, 1e3, N=4, as_tf=False)
    assert zpk.zeros.size == zpk.poles.size == 9
    assert np.all(zpk.zeros < 0) and np.all(zpk.poles < 0)  # stable, minimum phase


def test_oustaloup_returns_control_tf():
    tf = oustaloup(0.5, 1e-2, 1e2, N=3)
    assert isinstance(tf, control.TransferFunction)
    w = np.array([1.0])
    h_tf = tf(1j * w)
    h_zpk = freqresp_zpk(oustaloup(0.5, 1e-2, 1e2, N=3, as_tf=False), w)
    np.testing.assert_allclose(np.squeeze(h_tf), h_zpk[0], rtol=1e-8)


@pytest.mark.parametrize("kwargs", [dict(wb=0), dict(wb=10, wh=1), dict(N=0)])
def test_oustaloup_validation(kwargs):
    with pytest.raises(ValueError):
        oustaloup(0.5, **kwargs)


def test_zpk_to_tf_pure_gain():
    tf = ZPK(np.array([]), np.array([]), 2.0).to_tf()
    assert np.squeeze(tf(1j)) == pytest.approx(2.0)
