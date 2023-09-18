""" 
Helper function to deal with types of outputs.
"""

import numpy as np
from numpy.typing import ArrayLike


def conv_array(x: ArrayLike, is_array=False) -> np.ndarray:
    """
    Converts an array-like object to a np.ndarray if necessary.
    """
    if is_array:
        return x
    return np.array(x)
