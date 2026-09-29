from typing import Optional, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ..arrayfuncs import convert_to_ndarray
from .deviation_base import DeviationBaseClass

if TYPE_CHECKING:
    from ..bladerow import BladeRow  # for type hints only


class CarterDeviation(DeviationBaseClass):
    """Carter's rule for the exit deviation of axial compressor blades.

    With metal angles known (the default, from the row):
        deviation = (gamma_i - gamma_e) / (4 * sqrt(sigma))

    Design mode, when a target flow exit angle ``alpha_exit`` is given instead
    of ``gamma_exit`` (Mattingly's form of the same rule):
        gamma_e = (4 * alpha_exit * sqrt(sigma) - gamma_i) / (4 * sqrt(sigma) - 1)
        deviation = alpha_exit - gamma_e

    gamma_i/gamma_e are the inlet/exit metal angles and sigma the solidity. All
    angles are in degrees, and so is the result, as BladeRow expects from a
    deviation function.
    """

    def __init__(self, gamma_inlet: Optional[float | npt.ArrayLike] = None, gamma_exit: Optional[float | npt.ArrayLike] = None, alpha_exit: Optional[float | npt.ArrayLike] = None):
        """
        Args:
            gamma_inlet: Optional inlet metal angle (degrees), scalar or spanwise.
                Defaults to ``row.metal_inlet_angle``.
            gamma_exit: Optional exit metal angle (degrees), scalar or spanwise.
                Defaults to ``row.metal_exit_angle`` unless ``alpha_exit`` is given.
            alpha_exit: Optional target flow exit angle (degrees). When given
                without ``gamma_exit``, the exit metal angle is solved for.
        """
        super().__init__()
        self.gamma_inlet = gamma_inlet
        self.gamma_exit = gamma_exit
        self.alpha_exit = alpha_exit

    def _spanwise(self, value: float | npt.ArrayLike, row: "BladeRow") -> npt.NDArray:
        """Convert scalar/array input to spanwise distribution on row grid."""
        arr = convert_to_ndarray(value).astype(float).reshape(-1)
        if arr.size == 1:
            return arr * np.ones_like(row.percent_hub_shroud, dtype=float)
        t_src = np.linspace(0, 1, arr.size)
        return np.interp(row.percent_hub_shroud, t_src, arr)

    def __call__(self, row: "BladeRow", upstream: "BladeRow") -> npt.NDArray:  # noqa: ARG002
        """Compute deviation (degrees) along the span for the supplied row."""
        gamma_i = self.gamma_inlet if self.gamma_inlet is not None else row.metal_inlet_angle
        gamma_i_span = self._spanwise(gamma_i, row)
        k = np.sqrt(np.maximum(self._spanwise(row.solidity, row), 1e-6))

        if self.gamma_exit is None and self.alpha_exit is not None:
            alpha_exit_span = self._spanwise(self.alpha_exit, row)
            gamma_e_span = (4.0 * alpha_exit_span * k - gamma_i_span) / np.maximum(4.0 * k - 1.0, 1e-6)
            return alpha_exit_span - gamma_e_span

        gamma_e = self.gamma_exit if self.gamma_exit is not None else row.metal_exit_angle
        gamma_e_span = self._spanwise(gamma_e, row)
        return (gamma_i_span - gamma_e_span) / (4.0 * k)
