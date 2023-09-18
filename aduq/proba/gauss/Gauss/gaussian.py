"""
Subclass of Proba for Gaussian distributions

KL method and lin_transform are overwritten.
"""
import warnings
from typing import Iterable, List, Optional, Union

import numpy as np
from numpy.typing import ArrayLike

from ....misc import ShapeError
from ..._helper import _get_pre_shape, prod
from ..._types import SamplePoint, Samples
from ...proba import Proba
from ...warnings import NegativeCov


class Gaussian(Proba):
    """
    Gaussian multivariate distribution.
    Inherited from Proba class. Constructed from means and covariance matrix.

    Shape of samples:
        The shape of the samples can be specified from the sample_shape argument.
        If the sample_shape argument is None, this reverts to the shape of the means.
        If the sample_shape argument is incoherent with the shape of the means, throws a
            warning and uses the shape of the means.

        Regardless of the resulting sample_shape, the covariance must be a square matrix. If not,
        a warning is thrown and the covariance is reshaped if possible.

    Remark:
        The covariance/means should be immutable! Changing these attributes will have no effect
        whatsoever on the distribution and should NEVER be done (only adds confusion). Good
        practice is to generate a new Gaussian object.

        This behavior could be modified in the future by tracking changes to the covariance and
        checking for changes before using inv_cov/ eigenvalues/ eigenvectors. As it is, NEVER modify
        any attribute by hand!
    """

    def __init__(
        self,
        means: ArrayLike,
        cov: ArrayLike,
        info: Optional[dict] = None,
        sample_shape: Optional[tuple[int]] = None,
    ):
        """
        Constructs a gaussian distribution for mean and covariance matrix.

        Argument:
            means: means of the gaussian distribution.
            cov: covariance of the gaussian distribution.
            info: optional dictionary containing the eigenvalues and eigenvectors of the covariance (keys: 'vals', 'vects')
            sample_shape: optional tuple specifying the shape of the output. If not provided, use the shape of means.
        """

        # Force convert to np.ndarray
        means, cov = np.array(means), np.array(cov)

        # Check compatibility of means and sample_shape
        if sample_shape is None:
            sample_shape = means.shape
        else:
            if prod(sample_shape) == np.size(means):
                means = means.reshape(sample_shape)
            else:
                warnings.warn(
                    f"""Shape of means ({means.shape} and sample_shape ({sample_shape}) are not compatible.
                    Continuing using {means.shape} as sample_shape.
                    """
                )
                sample_shape = means.shape

        sample_dim = prod(sample_shape)

        # Check compatibility of means and covariance
        if (sample_dim**2) != prod(cov.shape):
            raise ShapeError(
                f"Covariance shape ({cov.shape}) and sample_shape ({sample_shape}) are not compatible."
            )

        # Check covariance format
        if len(cov.shape) != (sample_dim, sample_dim):
            # Try finding covariance by reshaping
            cov = cov.reshape(sample_dim, sample_dim)

        # Check that covariance is almost symmetric and force symmetry
        if np.allclose(cov, cov.T):
            cov = (cov + cov.T) / 2
        else:
            raise ValueError("'cov' must be symmetric")

        means = means.flatten()

        # From now on, one can assume that both cov and means are well formatted
        if info is None:
            vals, vects = np.linalg.eigh(cov)
        else:
            vals, vects = info["vals"], info["vects"]

        # Check for negative eigenvalues
        if vals[0] < 0:
            warnings.warn(
                "Covariance matrix had negative eigenvalues. Setting them to 0.",
                category=NegativeCov,
            )

        # Pre compute constants for log density function
        vals = np.array([val if val >= 0 else 0 for val in vals])

        singular = np.min(vals) == 0

        inv_vals = np.array([val**-1 if val > 0 else 0 for val in vals])
        inv_cov = (inv_vals * vects) @ vects.T

        renorm_const = -0.5 * sample_dim * np.log(2 * np.pi) - 0.5 * np.sum(
            np.log(vals)
        )

        # Define log_dens function for gaussian distribution
        def log_dens(samples: Samples) -> np.ndarray:
            samples = np.array(
                samples
            )  # Force convert array. Necessary due to transform behavior

            pre_shape = _get_pre_shape(samples, sample_shape)

            # Right flatten of array
            samples = samples.reshape(pre_shape + (sample_dim,))
            centered = samples - means
            # Using the fact that inv_cov is symmetric to use right multiplication
            dist = (centered * (centered @ inv_cov)).sum(-1)
            return -0.5 * dist + renorm_const

        # Pre compute constant for gen function
        half_vals = vals**0.5
        half_cov = (half_vals * vects) @ vects.T

        def gen(n: int) -> Iterable[SamplePoint]:
            # Very slightly more efficient than np.random.multivariate, notably if dim is large
            return (means + np.random.normal(0, 1, (n, sample_dim)) @ half_cov).reshape(
                (n,) + sample_shape
            )

        super().__init__(
            log_dens=log_dens, gen=gen, sample_shape=sample_shape, np_out=True
        )

        self.means = means
        self.shaped_means = means.reshape(sample_shape)
        self.cov = cov
        self.inv_cov = inv_cov
        self.singular = singular
        self.vects = vects
        self.vals = vals

    def __repr__(self):
        return str.join(
            "\n",
            [
                "Gaussian Distribution",
                f"Mean: {self.shaped_means}",
                f"Covariance : {self.cov}",
            ],
        )

    def copy(self):
        """
        Copy of the Gaussian distribution object.
        Avoids dependency on the cov and means.
        """
        return Gaussian(
            means=self.means.copy(),
            cov=self.cov.copy(),
            sample_shape=self.sample_shape,
            info={"vals": self.vals.copy(), "vects": self.vects.copy()},
        )

    def reshape(self, new_shape: tuple[int]):
        """
        Transforms the shape of the samples.

        If the distribution generates samples with each shapes of
            (n1, ..., nk),
        the distribution distr.reshape( (m1, ..., m \tilde{k})) will output samples of shape
            (m1, ..., m \tilde{k})
        IF n1 * ... * nk = m1 * ... * m \tilde{k} (else a ShapeMismatch exception is raised when
        trying to construct distr.reshape)

        Note:
            The new distribution will generate np.ndarray objects
        """
        return Gaussian(
            means=self.means,
            cov=self.cov,
            info={"vals": self.vals, "vects": self.vects},
            sample_shape=new_shape,
        )

    # flatten uses the .reshape method from the instance, no need to reimplement

    def contract(self, alpha: float):
        """
        Transform the distribution of X to the distribution of alpha * X

        Argument:
            alpha: a float
        """
        return Gaussian(
            means=alpha * self.means,
            cov=(alpha**2) * self.cov,
            info={"vals": (alpha**2) * self.vals, "vects": self.vects},
            sample_shape=self.sample_shape,
        )

    def shift(self, shift: SamplePoint):
        """
        Transform the distribution of X to the distribution of X + shift
        """
        return Gaussian(
            means=self.shaped_means + shift,
            cov=self.cov,
            info={"vals": self.vals, "vects": self.vects},
            sample_shape=self.sample_shape,
        )

    def lin_transform(self, mat: np.ndarray, shift: Union[float, SamplePoint] = 0.0):
        """
        Transform the distribution of X to the distribution of mat @ X + shift
        (where the @ denote a full tensor product rather than matrix product).

        Shift can be either a float or a np.array object of shape compatible with the matrix.

        Dimension formating:
            If distr outputs samples of shape (n1, ..., nk), then the matrix should be shaped
                (m1, ..., m \tilde{k}, n1, ..., nk)
            with m1, ..., m \tilde k such that m1 * ... * m\tilde{k} <= n1 * ... * nk.
            The new distribution will output samples of shape (m1, ..., m\tilde{k}).

        Note that contrary to the general behavior for lin_transform for the Proba class,
        this method allows the ouput dimension to be strictly smaller than the input dimension, as
        long as the covariance matrix has trivial kernel (to avoid renormalisation issue for the
        log-density)
        """

        n_dim_shape = len(self.sample_shape)
        n_dim_sample = prod(self.sample_shape)

        mat = np.array(mat)
        mat_shape = mat.shape

        if mat_shape[-n_dim_shape:] != self.sample_shape:
            raise Exception(
                f"The shape of the matrix should end with {self.sample_shape}"
            )

        if prod(mat_shape[:-n_dim_shape]) > n_dim_sample:
            raise Exception(
                f"The first {n_dim_shape} dimensions of the matrix should multiply to less than {n_dim_sample}"
            )

        new_shape = mat_shape[:-n_dim_shape]

        mat = mat.reshape((n_dim_sample, n_dim_sample))

        means = (mat @ self.means).reshape(new_shape) + shift
        new_cov = (mat @ self.cov) @ mat.T

        return Gaussian(means=means, cov=new_cov, sample_shape=new_shape)

    def marginalize(self, indexes: List[int]):
        """
        Get the marginal distribution of (X_i)_{i in indexes} from distribution of X.

        Due to renormalization issues in the general case, this method is only possible
        for specific probability distributions such as Gaussian.

        The resulting distribution operates on 1D array.
        """
        new_mean, new_cov = self.means.copy(), self.cov.copy()

        new_mean = new_mean[indexes]
        new_cov = new_cov[indexes]
        new_cov = new_cov[:, indexes]

        return Gaussian(new_mean, new_cov)
