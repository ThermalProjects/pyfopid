"""Rational (integer-order) approximations of fractional operators s^alpha.

Implements the Oustaloup recursive approximation

    s^g ~= K * prod_{k=-N}^{N} (s + w'_k) / (s + w_k),

    w'_k = wb * (wh / wb) ** ((k + N + (1 - g) / 2) / (2N + 1))
    w_k  = wb * (wh / wb) ** ((k + N + (1 + g) / 2) / (2N + 1))
    K    = wh ** g

valid inside the band [wb, wh] for |g| < 1 (Oustaloup et al., 2000,
IEEE Trans. Circuits Syst. I, 47(1), 25-39).

Orders with |alpha| >= 1 are split as alpha = n + g, with n = round(alpha) and
g in [-0.5, 0.5]; the integer part is realised exactly with zeros or poles at
the origin, and only g is approximated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["ZPK", "oustaloup", "freqresp_zpk"]


@dataclass(frozen=True)
class ZPK:
    """Zero-pole-gain representation: H(s) = gain * prod(s - z) / prod(s - p)."""

    zeros: NDArray[np.float64]
    poles: NDArray[np.float64]
    gain: float

    def to_tf(self):
        """Return a :class:`control.TransferFunction`."""
        import control

        num = self.gain * np.real_if_close(np.poly(self.zeros)) if self.zeros.size else np.array([self.gain])
        den = np.real_if_close(np.poly(self.poles)) if self.poles.size else np.array([1.0])
        return control.tf(np.atleast_1d(num).astype(float), np.atleast_1d(den).astype(float))


def oustaloup(
    alpha: float,
    wb: float = 1e-3,
    wh: float = 1e3,
    N: int = 5,
    as_tf: bool = True,
):
    """Oustaloup approximation of ``s**alpha`` in the band ``[wb, wh]`` rad/s.

    Parameters
    ----------
    alpha : float
        Fractional order (any real value).
    wb, wh : float
        Lower and upper band limits in rad/s, ``0 < wb < wh``.
    N : int
        Number of zero/pole pairs is ``2N + 1``.
    as_tf : bool
        If True return a :class:`control.TransferFunction`; otherwise a
        :class:`ZPK` object.
    """
    if not (wb > 0 and wh > wb):
        raise ValueError("require 0 < wb < wh")
    if not isinstance(N, (int, np.integer)) or N < 1:
        raise ValueError("N must be a positive integer")
    alpha = float(alpha)
    n_int = int(np.round(alpha))
    g = alpha - n_int

    zeros: list[float] = []
    poles: list[float] = []
    gain = 1.0
    if abs(g) > 1e-12:
        ratio = wh / wb
        k = np.arange(-N, N + 1)
        wz = wb * ratio ** ((k + N + 0.5 * (1.0 - g)) / (2 * N + 1))
        wp = wb * ratio ** ((k + N + 0.5 * (1.0 + g)) / (2 * N + 1))
        zeros.extend(-wz)
        poles.extend(-wp)
        gain = wh**g
    if n_int > 0:
        zeros.extend([0.0] * n_int)
    elif n_int < 0:
        poles.extend([0.0] * (-n_int))

    zpk = ZPK(np.array(zeros, dtype=float), np.array(poles, dtype=float), float(gain))
    return zpk.to_tf() if as_tf else zpk


def freqresp_zpk(zpk: ZPK, omega: ArrayLike) -> NDArray[np.complex128]:
    """Evaluate ``H(j*omega)`` of a :class:`ZPK` without forming polynomials."""
    s = 1j * np.asarray(omega, dtype=float)
    h = np.full(s.shape, zpk.gain, dtype=complex)
    for z in zpk.zeros:
        h *= s - z
    for p in zpk.poles:
        h /= s - p
    return h
