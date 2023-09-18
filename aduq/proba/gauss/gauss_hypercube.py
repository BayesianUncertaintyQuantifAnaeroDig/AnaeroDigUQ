"""
Pseudo Gaussian distribution map, with samples drawn in the hypercube of dimension sample_dim.

Function GaussHypercubeMap
"""
from typing import Optional

import numpy as np
from scipy.stats import norm

from .._helper import _get_pre_shape, _shape_info
from .._types import Samples
from ..proba_map import ProbaMap
from .Gauss import GaussianMap

normal = norm(0.0, 1.0)
cdf = normal.cdf
ppf = normal.ppf
pdf = normal.pdf


def GaussHypercubeMap(
    sample_dim: Optional[int] = None, sample_shape: Optional[tuple[int]] = None
) -> ProbaMap:
    """
    Pseudo Gaussian distribution map, with samples drawn in the hypercube of dimension sample_dim
    """
    sample_dim, sample_shape = _shape_info(sample_dim, sample_shape)

    def der_transform(xs: Samples) -> np.ndarray:
        pre_shape = _get_pre_shape(xs, sample_shape)
        return np.apply_along_axis(
            np.diag, -1, pdf(xs.reshape(pre_shape + (sample_dim,)))
        ).reshape(pre_shape + sample_shape + sample_shape)

    return GaussianMap(sample_dim=sample_dim).transform(
        transform=cdf, inv_transform=ppf, der_transform=der_transform
    )
