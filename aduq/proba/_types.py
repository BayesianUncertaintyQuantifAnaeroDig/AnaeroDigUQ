# from numpy.typing import ArrayLike


class ProbaParam:
    """
    Type hint for a parameter defining a probability distribution. Used for ProbaMap class.

    A ProbaParam should be convertible to a np.ndarray.
    """


# Type alias for a single sample
class SamplePoint:
    """
    Type hint class for a signle sample point

    A SamplePoint should be convertible to a np.ndarray of shape sample_shape
    """


class Samples:
    """
    Type hint class for multiple sample points stored properly in a np.ndarray object.

    The array should be of shape (pre_shape, sample_shape) where pre_shape may be anything,
    including (,).
    """
