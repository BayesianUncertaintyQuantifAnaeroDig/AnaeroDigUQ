import warnings
from typing import Callable, Optional

import numpy as np

from ..._helper import _get_pre_shape, _shape_info, prod
from ..._types import ProbaParam
from ...proba_map import ProbaMap
from ...warnings import ShapeWarning
from .gaussian import Gaussian


def make_cov(pre_cov: np.ndarray) -> np.ndarray:
    """Convert encoding of covariance to covariance. Used for GaussianMap"""
    return pre_cov @ pre_cov.T


def inverse_cov(cov: np.ndarray) -> np.ndarray:
    """Safe inversion of covariance. To do: modify and put in misc (can be useful in general)"""
    vals, vects = np.linalg.eigh(cov)
    inv_vals = np.array([val**-1 if val > 0 else 0 for val in vals])

    inv_cov = (inv_vals * vects) @ vects.T
    return inv_cov


class GaussianMap(ProbaMap):
    """
    For Gaussian distributions, use the following subclass (KL/grad_KL is overriden)

    The distribution param shape is (pred_param_len + 1, pred_param_len)
        - param[0] gives the mean,
        - param[1:] is a matrix m defining the covariance through $cov = m * m^T$

    The covariance is not used as a parameter to simplify routines such as gradient descents.

    sample_shape
    """

    def __init__(
        self,
        sample_dim: Optional[int] = None,
        sample_shape: Optional[tuple[int]] = None,
    ):
        """
        Define a GaussianMap on vectors of known shape.

        Either sample_dim or sample_shape must be specified. If both are, sample_dim is ignored.
        """

        sample_dim, sample_shape = _shape_info(sample_dim, sample_shape)

        def prob_map(x: ProbaParam) -> Gaussian:
            x = np.array(x)  # Force convert to array
            if x.shape != (sample_dim + 1, sample_dim):
                warnings.warn(
                    "\n".join(
                        [
                            f"Distribution parameter shape is {x.shape} (Expected{(sample_dim+1,sample_dim)})",
                            "Trying to construct nonetheless. Watch out for strange behaviour.",
                        ]
                    ),
                    category=ShapeWarning,
                )
            means = x[0]
            pre_cov = x[1:]
            cov = make_cov(pre_cov)
            return Gaussian(means=means, cov=cov, sample_shape=sample_shape)

        distr_param_shape = (sample_dim + 1, sample_dim)

        def log_dens_der(
            x: ProbaParam,
        ) -> Callable[[np.ndarray], np.ndarray]:
            means = x[0]
            pre_cov = x[1:]
            cov = make_cov(pre_cov)

            inv_cov = inverse_cov(cov)

            pre_compute_grad = -inv_cov @ pre_cov

            def derivative(samples: np.ndarray) -> np.ndarray:
                # ys is a np.ndarray with shape ending in sample shape
                pre_shape = _get_pre_shape(samples, sample_shape)

                pre_dim = prod(pre_shape)

                der = np.zeros((pre_dim,) + distr_param_shape)

                centered = (
                    samples.reshape(
                        (
                            pre_dim,
                            sample_dim,
                        )
                    )
                    - means
                )  # Shape (pre_dim, sample_dim)

                der_0 = centered @ inv_cov  # Shape (pre_dim, sample_dim)

                der[:, 0] = der_0

                # pre_compute_grad is of shape (sample_dim, sample_dim)
                # The np.outer stuff should be of shape (pre_dim, sample_dim, sample_dim)
                grad_param = (
                    pre_compute_grad
                    + np.apply_along_axis(lambda x: np.outer(x, x), -1, der_0) @ pre_cov
                )
                der[:, 1:] = grad_param

                return der.reshape(
                    pre_shape + distr_param_shape
                )  # Output in correct shape

            return derivative

        # Construct reference parameter (standard normal)
        ref_param = np.zeros((sample_dim + 1, sample_dim))
        ref_param[1:] = np.eye(sample_dim)

        super().__init__(
            prob_map=prob_map,
            log_dens_der=log_dens_der,
            ref_param=ref_param,
            distr_param_shape=(sample_dim + 1, sample_dim),
            sample_shape=sample_shape,
        )
        self.sample_dim = sample_dim

    def __repr__(self) -> str:
        return str.join(
            "\n", [f"Gaussian Prior Map on arrays of shape {self.sample_shape}."]
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
        if np.array_equal(param1, param0):
            return 0.0
        dim = self.sample_dim

        gauss1 = self.map(param1)
        gauss0 = self.map(param0)

        kl = np.sum(np.log(gauss0.vals)) - np.sum(np.log(gauss1.vals))
        kl = kl - dim + np.sum(np.diag(gauss0.inv_cov @ gauss1.cov))
        diff_means = gauss0.means - gauss1.means
        kl = kl + np.dot(diff_means, gauss0.inv_cov @ diff_means)
        kl = kl / 2
        if kl < 0:
            raise Exception(f"Negative KL ({kl}, numerical error)")
        return kl

    def grad_kl(
        self, param0: ProbaParam
    ) -> Callable[[ProbaParam, int], tuple[ProbaParam, float]]:
        """
        Approximates the gradient of the Kullback Leibler divergence between two distributions
        defined by their distribution parameters, with respect to the first distribution
        (nabla_{param1} KL(param1, param0))

        Args:
            param1, param0 are 2 distribution parameters
            n_sample is disregarded (exact computations used instead)

        Output:
            nabla_{param1}KL(param1, param0)
        """
        gauss0 = self.map(param0)

        def fun(param1: np.ndarray, n_sample: int = 0):  # pylint: disable=W0613
            if np.array_equal(param0, param1):
                return np.zeros(self.distr_param_shape), 0.0

            der = np.zeros(self.distr_param_shape)

            gauss1 = self.map(param1)
            pre_cov = param1[1:]

            der[0] = gauss0.inv_cov @ (gauss1.means - gauss0.means)
            grad_cov = 0.5 * (gauss0.inv_cov - gauss1.inv_cov)

            grad_param = 2 * grad_cov @ pre_cov
            der[1:] = grad_param

            return der, self.kl(param1, param0)

        return fun

    def grad_right_kl(
        self, param1: ProbaParam
    ) -> Callable[[ProbaParam, int], tuple[ProbaParam, float]]:
        gauss1 = self.map(param1)

        def fun(
            param0: ProbaParam,
            n_sample: int = 0,  # pylint: disable=W0613
        ) -> tuple[ProbaParam, float]:
            gauss0 = self.map(param0)

            der = np.zeros(self.distr_param_shape)

            der[0] = gauss0.inv_cov @ (gauss0.means - gauss1.means)
            grad_cov = 0.5 * (
                gauss0.inv_cov
                - gauss0.inv_cov @ gauss1.cov @ gauss0.inv_cov
                - np.outer(der[0], der[0])
            )

            pre_cov = param0[1:]
            grad_param = 2 * grad_cov @ pre_cov

            der[1:] = grad_param

            return der, self.kl(param1, param0)

        return fun

    def to_param(self, g_distr: Gaussian) -> np.ndarray:
        """
        Transforms a Gaussian back to a np.ndarray such that
        self.map(self.to_param(g_distr)) = g_distr
        """
        means = g_distr.means
        vals, vects = g_distr.vals, g_distr.vects
        accu = np.zeros(self.distr_param_shape)
        accu[0] = means
        accu[1:] = np.sqrt(vals) * vects
        return accu
