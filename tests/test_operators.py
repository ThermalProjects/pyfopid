import math

import numpy as np
import pytest
from scipy.special import binom, gamma

from pyfopid import caputo_derivative, fractional_integral, gl_derivative, gl_weights


# --- Weights ----------------------------------------------------------------

@pytest.mark.parametrize("alpha", [-1.5, -0.7, 0.3, 0.5, 0.9, 1.4, 2.0])
def test_weights_match_signed_binomial(alpha):
    n = 30
    j = np.arange(n)
    expected = (-1.0) ** j * binom(alpha, j)
    np.testing.assert_allclose(gl_weights(alpha, n), expected, rtol=1e-12, atol=1e-15)


def test_weights_sign_pattern_derivative():
    # Regression guard: for 0 < alpha < 1 the first weight is +1 and all others
    # are negative, i.e. (-1)^j * binom(alpha, j) is correctly signed.
    w = gl_weights(0.6, 50)
    assert w[0] == 1.0
    assert np.all(w[1:] < 0)


def test_weights_integer_orders():
    np.testing.assert_array_equal(gl_weights(0.0, 4), [1, 0, 0, 0])
    np.testing.assert_array_equal(gl_weights(1.0, 4), [1, -1, 0, 0])
    np.testing.assert_array_equal(gl_weights(2.0, 4), [1, -2, 1, 0])
    np.testing.assert_array_equal(gl_weights(-1.0, 4), [1, 1, 1, 1])


def test_weights_invalid_n():
    with pytest.raises(ValueError):
        gl_weights(0.5, 0)


# --- GL derivative against analytic results -------------------------------

def analytic_power(k, alpha, t):
    """D^alpha t^k = Gamma(k+1) / Gamma(k+1-alpha) * t^(k-alpha)."""
    return gamma(k + 1) / gamma(k + 1 - alpha) * t ** (k - alpha)


@pytest.mark.parametrize("alpha", [0.3, 0.5, 0.8, 1.5, -0.5, -1.0])
def test_gl_power_function(alpha):
    h = 1e-4
    t = np.arange(0, 1 + h / 2, h)
    f = t**2
    y = gl_derivative(f, alpha, h)
    ref = analytic_power(2, alpha, t[-1])
    assert abs(y[-1] - ref) / abs(ref) < 5e-4


def test_gl_first_order_convergence():
    alpha = 0.5
    errs = []
    for h in [1e-2, 5e-3, 2.5e-3]:
        t = np.arange(0, 1 + h / 2, h)
        y = gl_derivative(t**2, alpha, h)
        errs.append(abs(y[-1] - analytic_power(2, alpha, 1.0)))
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    np.testing.assert_allclose(orders, 1.0, atol=0.1)


def test_gl_integer_derivative_is_backward_difference():
    rng = np.random.default_rng(0)
    f = rng.normal(size=100)
    h = 0.1
    y = gl_derivative(f, 1.0, h)
    np.testing.assert_allclose(y[1:], np.diff(f) / h, rtol=1e-12)


def test_gl_identity():
    f = np.linspace(-1, 3, 50)
    np.testing.assert_allclose(gl_derivative(f, 0.0, 0.01), f)


def test_short_memory_converges_to_full():
    h = 1e-3
    t = np.arange(0, 2 + h / 2, h)
    f = np.sin(t)
    full = gl_derivative(f, 0.5, h)
    errs = [abs(gl_derivative(f, 0.5, h, memory=L)[-1] - full[-1]) for L in (200, 800, 1600)]
    assert errs[0] > errs[1] > errs[2]
    assert gl_derivative(f, 0.5, h, memory=len(f))[-1] == pytest.approx(full[-1])


def test_fractional_integral_of_constant():
    # I^a 1 = t^a / Gamma(1 + a)
    a, h = 0.5, 1e-4
    t = np.arange(0, 1 + h / 2, h)
    y = fractional_integral(np.ones_like(t), a, h)
    ref = 1.0 / gamma(1 + a)
    assert abs(y[-1] - ref) / ref < 1e-3


def test_fractional_integral_rejects_negative_order():
    with pytest.raises(ValueError):
        fractional_integral([1, 2, 3], -0.5, 0.1)


# --- Caputo -------------------------------------------------------------------

def test_caputo_of_constant_is_zero():
    y = caputo_derivative(np.full(200, 3.0), 0.7, 0.01)
    np.testing.assert_allclose(y, 0.0, atol=1e-12)


def test_caputo_vs_rl_for_shifted_power():
    # f = 1 + t^2: Caputo removes the constant, RL does not.
    alpha, h = 0.5, 1e-4
    t = np.arange(0, 1 + h / 2, h)
    y = caputo_derivative(1 + t**2, alpha, h)
    ref = analytic_power(2, alpha, 1.0)
    assert abs(y[-1] - ref) / ref < 5e-4


def test_caputo_order_above_one_requires_initial():
    t = np.linspace(0, 1, 101)
    with pytest.raises(ValueError):
        caputo_derivative(1 + t, 1.5, 0.01)
    # f = 1 + t + t^2 with f(0)=1, f'(0)=1 -> C D^1.5 f = D^1.5 t^2
    h = 1e-4
    t = np.arange(0, 1 + h / 2, h)
    y = caputo_derivative(1 + t + t**2, 1.5, h, initial=[1.0, 1.0])
    ref = analytic_power(2, 1.5, 1.0)
    assert abs(y[-1] - ref) / ref < 1e-3


def test_caputo_rejects_non_positive_order():
    with pytest.raises(ValueError):
        caputo_derivative([1, 2, 3], 0.0, 0.1)


def test_input_validation():
    with pytest.raises(ValueError):
        gl_derivative([1, 2, 3], 0.5, 0.0)
    with pytest.raises(ValueError):
        gl_derivative([[1, 2], [3, 4]], 0.5, 0.1)
    assert gl_derivative([], 0.5, 0.1).size == 0
    assert math.isfinite(gl_derivative([1.0], 0.5, 0.1)[0])
