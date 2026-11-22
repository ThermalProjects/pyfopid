"""pyfopid: fractional-order modelling and control in Python."""

from .approx import ZPK, freqresp_zpk, oustaloup
from .controller import FOPID
from .operators import caputo_derivative, fractional_integral, gl_derivative, gl_weights

__version__ = "0.1.0.dev0"

__all__ = [
    "FOPID",
    "ZPK",
    "caputo_derivative",
    "fractional_integral",
    "freqresp_zpk",
    "gl_derivative",
    "gl_weights",
    "oustaloup",
    "__version__",
]
