"""
Special GaussianMaps when the covariance is either fixed or fixed up to a factor.

Rationale for the classes:
    - Standard gaussian map heavily relies on covariance matrix construction/inversion
    when computing KL related quantiites. In this setting, this information can be predetermined.
"""

from typing import Callable, Optional

import numpy as np
from numpy.typing import ArrayLike

from ...misc import ShapeError
from .._helper import _get_pre_shape, prod
from .._types import ProbaParam, Samples
from ..proba_map import ProbaMap
from .Gauss import Gaussian, inverse_cov


class FactCovGaussianMap(ProbaMap):
    r"""
    Parametrization of gaussian distributions of form
            $(mu, sigma) -> N(mu, sigma^2 * Cov)$
    where Cov is fixed.
    """

    def __init__(
        self,
        sample_dim: Optional[int] = None,
        sample_shape: Optional[tuple[int]] = None,
        cov: Optional[ArrayLike] = None,
    ):

        if (sample_dim is None) and (sample_shape is None):
            raise Exception("Either sample_dim or sample_shape must be specified.")

        if sample_shape is None:
            sample_shape = (sample_dim,)

        sample_dim = prod(
            sample_shape
        )  # Define if sample_dim is missing/Force coherence if both are specified

        if cov is None:
            cov = np.eye(sample_dim)
        else:
            cov = np.array(cov)
            if cov.shape != (sample_dim, sample_dim):
                raise Exception(
                    f"Expected a covariance matrix of shape {(sample_dim, sample_dim)}"
                )

        inv_cov = inverse_cov(cov)  # pre compute it once and for all

        def prob_map(x: ProbaParam) -> Gaussian:
            x = np.array(x)  # Force convert to array

            if x.shape != (sample_dim + 1,):
                raise ShapeError(
                    f"Distribution parameter shape is {x.shape} (Expected{(sample_dim+1,)})"
                )

            return Gaussian(
                means=x[:-1], cov=(x[-1] ** 2) * cov, sample_shape=sample_shape
            )

        def log_dens_der(x: ProbaParam) -> Callable[[Samples], np.ndarray]:
            """Signature: ProbaParam -> (Sample -> ProbaParam)"""
            x = np.array(x)
            means, sigma = x[:-1], x[-1]

            def derivative(samples: Samples) -> np.ndarray:
                pre_shape = _get_pre_shape(samples, sample_shape)

                centered = samples.reshape(pre_shape + (sample_dim,)) - means
                der_means = (centered @ inv_cov) * sigma ** (-2)

                der_sigma = sigma ** (-1) * (
                    (centered * der_means).sum(-1) - sample_dim
                )

                res = np.zeros((prod(pre_shape), sample_dim + 1))
                res[:, :-1] = der_means.flatten()
                res[:, -1] = der_sigma.flatten()

                return res.reshape(pre_shape + (sample_dim + 1,))

            return derivative

        ref_param = np.zeros(sample_dim + 1)
        ref_param[-1] = 1

        super().__init__(
            prob_map=prob_map,
            log_dens_der=log_dens_der,
            ref_param=ref_param,
            distr_param_shape=(sample_dim + 1,),
            sample_shape=sample_shape,
        )
        self.sample_dim = sample_dim
        self.cov = cov
        self.inv_cov = inv_cov

    def __repr__(self) -> str:
        return str.join(
            "\n",
            [
                f"Gaussian Prior Map with covariance fixed up to a factor a on arrays of shape {self.sample_shape}.",
                f"Default covariance:\n{self.cov}",
            ],
        )

    def kl(self, param1: np.ndarray, param0: np.ndarray, n_sample: int = 0) -> float:
        """Computes the Kullback Leibler divergence between two gaussian distributions.
        defined by their prior parameters.

        Args:
            distrib1, distrib0 are 2 prior parameters
            n_sample is disregarded (exact computations used instead)

        Output:
            kl(distrib1, distrib0)

        .5 * (2k * log(sigma0/sigma1) - k + k (sigma1/sigma0) **2 + sigma0 ** (-2) diff_means * inv_cov @ diff_means
        """

        diff_means = param0[:-1] - param1[:-1]
        sigma0, sigma1 = param0[-1], param1[-1]
        ratio_sig = (sigma1 / sigma0) ** 2
        kl = (
            sigma0 ** (-2) * np.dot(diff_means, self.inv_cov @ diff_means)
            + (-np.log(ratio_sig) - 1 + ratio_sig) * self.sample_dim
        ) / 2

        return kl

    def grad_kl(
        self, param0: np.ndarray
    ) -> Callable[[np.ndarray, int, bool], np.ndarray]:
        """
        Approximates the gradient of the Kullback Leibler divergence between two distributions
        defined by their distribution parameters, with respect to the first distribution
        (nabla_{param1} kl(param1, param0))

        Args:
            param1, param0 are 2 distribution parameters
            n_sample is disregarded (exact computations used instead)

        Output:
            nabla_{param1}kl(param1, param0)
            See doc for more information.
        """
        inv_cov = self.inv_cov
        means0 = param0[:-1]
        sigma0 = param0[-1]

        def fun(
            param1: np.ndarray,
            n_sample: int = 0,  # pylint: disable=W0613
        ) -> np.ndarray:
            diff_mean = param1[:-1] - means0
            sigma1 = param1[-1]

            grad_kl_mean = sigma0 ** (-2) * inv_cov @ diff_mean

            ratio_sig = (sigma1 / sigma0) ** 2
            grad_ratio_sig = 2 * ratio_sig / sigma1

            grad_kl_sig = grad_ratio_sig * (-1 / ratio_sig + 1) * self.sample_dim

            der = np.zeros(self.sample_dim + 1)
            der[:-1] = grad_kl_mean
            der[-1] = 0.5 * grad_kl_sig

            return der, 0.5 * (
                np.dot(diff_mean, grad_kl_mean)
                + (-np.log(ratio_sig) - 1 + ratio_sig) * self.sample_dim
            )

        return fun

    def grad_right_kl(
        self, param1: np.ndarray
    ) -> Callable[[np.ndarray, int], np.ndarray]:
        """
        Approximates the gradient of the Kullback Leibler divergence between two distributions
        defined by their distribution parameters, with respect to the first distribution
        (nabla_{param1} kl(param1, param0))

        Args:
            param1, param0 are 2 distribution parameters
            n_sample is disregarded (exact computations used instead)

        Output:
            nabla_{param1}kl(param1, param0)
            See doc for more information.
        """
        inv_cov = self.inv_cov
        means1 = param1[:-1]
        sigma1 = param1[-1]

        def fun(
            param0: np.ndarray,
            n_sample: int = 0,  # pylint: disable=W0613
        ) -> np.ndarray:
            diff_mean = param0[:-1] - means1
            sigma0 = param0[-1]

            grad_kl_mean = sigma0 ** (-2) * inv_cov @ diff_mean

            ratio_sig = (sigma1 / sigma0) ** 2
            grad_ratio_sig = -2 * ratio_sig / sigma0

            grad_kl_sig = grad_ratio_sig * (-1 / ratio_sig + 1) * self.sample_dim

            der = np.zeros(self.sample_dim + 1)
            der[:-1] = grad_kl_mean
            der[-1] = 0.5 * grad_kl_sig - np.dot(diff_mean, grad_kl_mean) / sigma0

            return der, 0.5 * (
                np.dot(diff_mean, grad_kl_mean)
                + (-np.log(ratio_sig) - 1 + ratio_sig) * self.sample_dim
            )

        return fun


class FixedCovGaussianMap(ProbaMap):
    r"""
    Class for Gaussian probability distributions of form $\mathcal{N}(\mu, \Sigma)$ with $\Sigma$
    fixed.
    """

    def __init__(
        self,
        sample_dim: Optional[int] = None,
        sample_shape: Optional[tuple[int]] = None,
        cov: Optional[ArrayLike] = None,
    ):

        if (sample_dim is None) and (sample_shape is None):
            raise Exception("Either sample_dim or sample_shape must be specified.")

        if sample_shape is None:
            sample_shape = (sample_dim,)

        sample_dim = prod(
            sample_shape
        )  # Define if sample_dim is missing/Force coherence if both are specified

        if cov is None:
            cov = np.eye(sample_dim)
        else:
            cov = np.array(cov)
            if cov.shape != (sample_dim, sample_dim):
                raise Exception(
                    f"Expected a covariance matrix of shape {(sample_dim, sample_dim)}"
                )

        inv_cov = inverse_cov(cov)  # pre compute it once and for all

        def prob_map(x: ProbaParam) -> Gaussian:
            x = np.array(x)  # Force convert to array
            if x.shape != (sample_dim,):
                raise Exception(
                    "\n".join(
                        [
                            f"Distribution parameter shape is {x.shape} (Expected{(sample_dim,)})",
                            "Trying to construct nonetheless. Watch out for strange behaviour.",
                        ]
                    )
                )

            return Gaussian(means=x, cov=cov, sample_shape=sample_shape)

        def log_dens_der(x: ProbaParam) -> Callable[[Samples], np.ndarray]:
            means = x

            def derivative(samples: Samples) -> np.ndarray:
                pre_shape = _get_pre_shape(samples, sample_shape)
                centered = samples.reshape((pre_shape) + (sample_dim,)) - means
                return centered @ inv_cov

            return derivative

        super().__init__(
            prob_map=prob_map,
            log_dens_der=log_dens_der,
            ref_param=np.zeros(sample_dim),
            distr_param_shape=(sample_dim,),
            sample_shape=sample_shape,
        )
        self.sample_dim = sample_dim
        self.cov = cov
        self.inv_cov = inv_cov

    def __repr__(self) -> str:
        return str.join(
            "\n",
            [
                f"Gaussian Prior Map with fixed covariance on arrays of shape {self.sample_shape}.",
                f"Covariance:\n{self.cov}",
            ],
        )

    def kl(self, param1: np.ndarray, param0: np.ndarray, n_sample: int = 0) -> float:
        """Computes the Kullback Leibler divergence between two gaussian distributions.
        defined by their prior parameters.

        Args:
            distrib1, distrib0 are 2 prior parameters
            n_sample is disregarded (exact computations used instead)

        Output:
            kl(distrib1, distrib0)
        """

        diff_means = param0 - param1
        kl = np.dot(diff_means, self.inv_cov @ diff_means)
        kl = kl / 2
        return kl

    def grad_kl(self, param0: np.ndarray) -> Callable[[np.ndarray, int], np.ndarray]:
        """
        Approximates the gradient of the Kullback Leibler divergence between two distributions
        defined by their distribution parameters, with respect to the first distribution
        param0 ->(param1  ->  (nabla_{param1} kl(param1, param0)))

        Args:
            param0 is a parameter describing the right distribution.
        Output:
            nabla_{param1}kl(param1, param0)
            See doc for more information.
        """

        inv_cov = self.inv_cov

        def fun(
            param1: np.ndarray,
            n_sample: int = 0,  # pylint: disable=W0613
        ):
            diff_mean = param1 - param0
            grad_kl = inv_cov @ diff_mean
            return grad_kl, 0.5 * np.dot(diff_mean, grad_kl)

        return fun

    def grad_right_kl(
        self, param1: ProbaParam
    ) -> Callable[[ProbaParam, int], tuple[ProbaParam, float]]:
        """
        Approximates the gradient of the Kullback Leibler divergence between two distributions
        defined by their distribution parameters, with respect to the second distribution
        (nabla_{param0} kl(param1, param0))

        Args:
            param1, param0 are 2 distribution parameters
            n_sample is disregarded (exact computations used instead)

        Output:
            nabla_{param1}kl(param1, param0)
            See doc for more information.
        """
        inv_cov = self.inv_cov

        def fun(
            param0: ProbaParam,
            n_sample: int = 0,  # pylint: disable=W0613
        ) -> tuple[ProbaParam, float]:
            diff_mean = param0 - param1
            grad_kl = inv_cov @ diff_mean
            return grad_kl, 0.5 * np.dot(diff_mean, grad_kl)

        return fun

    def to_param(self, g_distr: Gaussian) -> np.ndarray:
        """
        Transforms a Gaussian back to a np.ndarray such that
        self.map(self.to_param(g_distr)) = g_distr
        """
        return g_distr.means
