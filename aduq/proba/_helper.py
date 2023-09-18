"""Miscallenous helper function for Gaussian distributions/Maps"""
from typing import Optional

import numpy as np

from ..misc import ShapeError
from .warnings import ShapeWarning


def prod(x: tuple[int]) -> int:
    """Minor correction to np.prod function in the case where the shape is ()"""
    return int(np.prod(x))


def _shape_info(
    sample_dim: Optional[int] = None, sample_shape: Optional[tuple[int]] = None
) -> tuple[int, tuple[int]]:
    if (sample_dim is None) and (sample_shape is None):
        raise ValueError("Either 'sample_dim' or 'sample_shape' must be specified.")

    if sample_shape is None:
        sample_shape = (sample_dim,)

    elif sample_dim is None:
        sample_dim = prod(
            sample_shape
        )  # Define if sample_dim is missing/Force coherence if both are specified

    elif sample_dim != prod(sample_shape):
        ShapeWarning(
            f"'sample_dim' {sample_dim} and 'sample_shape' {sample_shape} arguments are incoherent. Using 'sample_shape' information"
        )
        sample_dim = prod(sample_shape)

    return sample_dim, sample_shape


def _get_pre_shape(xs: np.ndarray, exp_shape: tuple[int]) -> tuple[int]:
    if exp_shape == ():
        return xs.shape
    n_dim = len(exp_shape)
    tot_shape = xs.shape

    if len(tot_shape) < n_dim:
        raise ShapeError("Shape of input array is not compliant with expected shape")

    if tot_shape[-n_dim:] != exp_shape:
        raise ShapeError("Shape of input array is not compliant with expected shape")

    return tot_shape[:-n_dim]
