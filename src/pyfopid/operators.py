"""Fractional-order differ-integral operators on uniformly sampled signals.

All operators use the Grünwald-Letnikov (GL) definition

    D^alpha f(t_k) ~= h^(-alpha) * sum_{j=0}^{k} w_j f(t_{k-j}),

with the binomial weights w_j = (-1)^j * binom(alpha, j), computed with the
numerically stable recursion

    w_0 = 1,    w_j = w_{j-1} * (1 - (alpha + 1) / j).

The recursion produces the alternating sign pattern of (-1)^j automatically.
Positive ``alpha`` gives a derivative, negative ``alpha`` an integral and
``alpha = 0`` the identity.

For functions with f(0) = 0 (and sufficiently smooth), the GL, Riemann-Liouville
and Caputo derivatives coincide. The GL scheme is first-order accurate, O(h).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "gl_weights",
    "gl_derivative",
    "fractional_integral",
    "caputo_derivative",
]


def gl_weights(alpha: float, n: int) -> NDArray[np.float64]:
    """Return the first ``n`` Grünwald-Letnikov weights for order ``alpha``.

    Parameters
    ----------
    alpha : float
        Order of the operator. Positive for derivatives, negative for integrals.
    n : int
        Number of weights (memory length + 1). Must be >= 1.

    Returns
    -------
    numpy.ndarray
        Array ``w`` with ``w[j] = (-1)**j * binom(alpha, j)``.

    Examples
    --------
    >>> gl_weights(2.0, 3)
    array([ 1., -2.,  1.])
    >>> gl_weights(-1.0, 3)
    array([1., 1., 1.])
    """
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    alpha = float(alpha)
    w = np.empty(int(n), dtype=np.float64)
    w[0] = 1.0
    for j in range(1, int(n)):
        w[j] = w[j - 1] * (1.0 - (alpha + 1.0) / j)
    return w


def gl_derivative(
    f: ArrayLike,
    alpha: float,
    h: float,
    memory: int | None = None,
) -> NDArray[np.float64]:
    """Grünwald-Letnikov fractional derivative (or integral) of a sampled signal.

    Parameters
    ----------
    f : array_like
        Samples ``f(t_0), f(t_1), ...`` on a uniform grid starting at ``t_0``.
    alpha : float
        Order. ``alpha > 0`` derivative, ``alpha < 0`` integral.
    h : float
        Sampling step (> 0).
    memory : int, optional
        Short-memory length ``L`` (number of past samples used). ``None`` uses
        the full history (exact GL sum). Truncation introduces an error that
        decays as ``L**(-alpha)`` for derivatives.

    Returns
    -------
    numpy.ndarray
        ``D^alpha f`` evaluated at every sample, same length as ``f``.
    """
    f = np.asarray(f, dtype=np.float64)
    if f.ndim != 1:
        raise ValueError("f must be one-dimensional")
    if h <= 0:
        raise ValueError("h must be positive")
    n = f.size
    if n == 0:
        return f.copy()
    if memory is None:
        n_w = n
    else:
        if memory < 0:
            raise ValueError("memory must be >= 0")
        n_w = min(int(memory) + 1, n)
    w = gl_weights(alpha, n_w)
    y = np.convolve(f, w)[:n]
    return y * h ** (-float(alpha))


def fractional_integral(
    f: ArrayLike,
    order: float,
    h: float,
    memory: int | None = None,
) -> NDArray[np.float64]:
    """Fractional integral of order ``order > 0`` (GL / Riemann-Liouville).

    Equivalent to ``gl_derivative(f, -order, h, memory)``.
    """
    if order < 0:
        raise ValueError("order must be non-negative; use gl_derivative for derivatives")
    return gl_derivative(f, -float(order), h, memory)


def caputo_derivative(
    f: ArrayLike,
    alpha: float,
    h: float,
    initial: ArrayLike | None = None,
    memory: int | None = None,
) -> NDArray[np.float64]:
    """Caputo fractional derivative of order ``alpha > 0``.

    Uses the identity

        C D^alpha f = RL D^alpha [ f - sum_{k=0}^{m-1} f^(k)(0) t^k / k! ],

    with ``m = ceil(alpha)``, and evaluates the right-hand side with the GL sum.

    Parameters
    ----------
    f : array_like
        Uniform samples starting at ``t = 0``.
    alpha : float
        Order (> 0).
    h : float
        Sampling step.
    initial : array_like, optional
        Initial derivatives ``(f(0), f'(0), ..., f^(m-1)(0))``. For
        ``0 < alpha <= 1`` it can be omitted and ``f[0]`` is used. For
        ``alpha > 1`` it is required.
    memory : int, optional
        Short-memory length, see :func:`gl_derivative`.
    """
    alpha = float(alpha)
    if alpha <= 0:
        raise ValueError("Caputo derivative requires alpha > 0")
    f = np.asarray(f, dtype=np.float64)
    m = math.ceil(alpha)
    if initial is None:
        if m > 1:
            raise ValueError(
                f"alpha={alpha} needs {m} initial derivatives; pass `initial`"
            )
        initial_arr = np.array([f[0]])
    else:
        initial_arr = np.atleast_1d(np.asarray(initial, dtype=np.float64))
        if initial_arr.size != m:
            raise ValueError(f"`initial` must have length ceil(alpha) = {m}")
    t = np.arange(f.size) * h
    taylor = np.zeros_like(f)
    for k, dk in enumerate(initial_arr):
        taylor += dk * t**k / math.factorial(k)
    return gl_derivative(f - taylor, alpha, h, memory)
