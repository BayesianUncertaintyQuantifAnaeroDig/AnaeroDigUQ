import warnings
from typing import Iterable, Optional, Union

import numpy as np

from ..._helper import prod
from ..._types import SamplePoint, Samples
from ...proba import Proba
from ..Gauss import Gaussian


class TensorizedGaussian(Proba):
    """
    Gaussian multivariate distribution with diagonal covariance matrix class.
    Inherited from Proba class. Constructed from means and standard deviations.

    As for Proba, it is assumed that the distribution is defined on np.ndarray like.
    The shape of a sample is defined by the shape of the mean. Stored information is
    flattened.
    """

    def __init__(
        self,
        means: np.ndarray,
        devs: np.ndarray,
        sample_shape: Optional[tuple[int]] = None,
    ):
        """
        Constructs a gaussian distribution from means and standard deviations, assuming
        that the covariance is diagonal.
        The shape of the sample is determined by the shape of the means. The standard
        deviations is flattened.
        """

        means, devs = np.array(means), np.array(devs)

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

        n_dim = prod(sample_shape)
        n_dim_shape = len(sample_shape)

        means, devs = means.flatten(), devs.flatten()

        if len(devs) != len(means):
            raise ValueError("Means and standard deviations should have the same size")
            # To Do: Case where dev is a float (i.e. one dev for all)

        renorm_const = -0.5 * n_dim * np.log(2 * np.pi) - np.sum(np.log(devs))

        def log_dens(samples: Samples) -> np.ndarray:
            samples = np.array(samples)
            pre_shape = samples.shape[:-n_dim_shape]
            samples = samples.reshape(pre_shape + (n_dim,))

            return -0.5 * (((samples - means) / devs) ** 2).sum(-1) + renorm_const

        def gen(n: int) -> Iterable[SamplePoint]:
            return (means + devs * np.random.normal(0, 1, (n, n_dim))).reshape(
                (n,) + sample_shape
            )

        super().__init__(
            log_dens=log_dens, gen=gen, sample_shape=sample_shape, np_out=True
        )

        self.means = means
        self.devs = devs

    def __repr__(self):
        return str.join(
            "\n",
            [
                f"Means:\n{self.means.reshape(self.sample_shape)}\n",
                f"Standard deviations:\n{self.devs.reshape(self.sample_shape)}",
            ],
        )

    def copy(self):
        return TensorizedGaussian(self.means.copy(), self.devs.copy())

    def as_Gaussian(self) -> Gaussian:
        return Gaussian(
            means=self.means.copy(),
            cov=np.diag(self.devs**2),
            info={"vals": self.devs**2, "vects": np.eye(len(self.devs))},
        )

    def shift(self, shift: SamplePoint):
        """
        Transform the distribution of X to the distribution of X + shift
        """
        return TensorizedGaussian(
            means=self.means.reshape(self.sample_shape) + shift,
            devs=self.devs,
            sample_shape=self.sample_shape,
        )

    def contract(self, alpha: float):
        """
        Transform the distribution of X to the distribution of alpha * X

        Argument:
            alpha: a float
        """
        return TensorizedGaussian(
            means=self.means,
            devs=alpha * self.devs,
            sample_shape=self.sample_shape,
        )

    def lin_transform(self, mat: np.ndarray, shift: Union[float, SamplePoint] = 0.0):
        """
        Transform the distribution of X to the distribution of mat @ X + shift
        (where the @ denote a full tensor product rather than matrix product).

        Shift can be either a float or a np.array object of shape compatible with the matrix.
        """

        return self.as_Gaussian().lin_transform(mat=mat, shift=shift)

    def marginalize(self, indexes: list[int]):
        """
        Get the marginal distribution of (X_i)_{i in indexes} from distribution of X.

        Due to renormalization issues in the general case, this method is only possible
        for specific probability distributions such as Gaussian.

        The resulting distribution operates on 1D array.
        """
        new_means, new_devs = self.means.copy(), self.devs.copy()

        new_means = new_means[indexes]
        new_devs = new_devs[indexes]

        return TensorizedGaussian(new_means, new_devs, sample_shape=(len(new_means),))
