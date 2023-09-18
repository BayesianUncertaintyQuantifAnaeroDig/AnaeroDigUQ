"""
Default way to interpret a free parameter.

The transformation used is a shifted log transform component wise.
"""

import numpy as np

from ..misc import interpretation
from ._typing import DigesterParameter

# Default values shared between optimisation methods
default_param = np.array(
    [
        1.2,  # mu1max, +- .1, d^{-1}
        0.74,  # mu2max, +- .16, d^{-1}
        7.1,  # KS1, +- .08, gCOD L^{-1}
        9.28,  # KS2, +- 3.62, mmol L^{-1}
        256.0,  # KI2, +- 76.14, mmol L^{-1}
    ]
)

# Parameter interpretation routine
def par_map(par: np.ndarray) -> DigesterParameter:
    return np.exp(par) * default_param


def inv_par_map(par: DigesterParameter) -> np.ndarray:
    return np.log(par / default_param)


interp_param = interpretation(par_map)
