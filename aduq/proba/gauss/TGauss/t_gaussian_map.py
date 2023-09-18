"""
Map for Gaussian distributions with diagonal covariance.
"""

import warnings
from typing import Callable, Optional

import numpy as np

from ....misc import ShapeError
from ..._helper import _get_pre_shape, _shape_info, prod
from ..._types import ProbaParam, Samples
from ...proba_map import ProbaMap
from .t_gaussian import TensorizedGaussian


class TensorizedGaussianMap(ProbaMap):
    """
    For Gaussian tensorized distributions, use the following subclass (kl and grad_kl are overriden)
    The distribution param shape is (2, sample_shape)
    The first element is the mean, the second element controls the standard deviation through:
    sigma = np.abs(x)
    Note that this class could be reimplemented using the reparametrize from GaussianMap.
    This implementation is slightly more efficient as it takes advantage of the fact that the
    covriance is diagonal.
    """

    def __init__(
        self,
        sample_dim: Optional[int] = None,
        sample_shape: Optional[tuple[int]] = None,
    ):
        """
         Construct the family of gaussian distributions with independant components (tensorized gaussians), from the shape of the sample.

        Either sample_dim or sample_shape must be specified. If both are, sample_dim is ignored.

        The resulting family is parametrized by objects of shape  (2, sample_shape),
        the first element being the mean, the second element controling the standard deviation through:
                sigma = np.abs(x)
        """

        sample_dim, sample_shape = _shape_info(sample_dim, sample_shape)

        distr_param_shape = (2, sample_dim)

        def prob_map(x: ProbaParam) -> TensorizedGaussian:
            x = np.array(x)
            if x.shape != distr_param_shape:
                warnings.warn(
                    "\n".join(
                        [
                            f"Distribution parameter shape is {x.shape} (Expected{distr_param_shape})",
                            "Trying to construct nonetheless. Watch out for strange behaviour.",
                        ]
                    )
                )

            return TensorizedGaussian(
                means=x[0], devs=np.abs(x[1]), sample_shape=sample_shape
            )

        def log_dens_der(x: ProbaParam) -> Callable[[Samples], np.ndarray]:
            x = np.array(x)
            means = x[0]
            signed_devs = x[1]
            inv_var = signed_devs ** (-2)

            def derivate(samples: Samples) -> np.ndarray:
                pre_shape = _get_pre_shape(samples, sample_shape)
                der = np.zeros((prod(pre_shape),) + distr_param_shape)
                centered = samples.reshape((prod(pre_shape), sample_dim)) - means
                der[:, 0] = centered * inv_var
                der[:, 1] = -(1 - (centered**2) * inv_var) / signed_devs
                return der.reshape(pre_shape + distr_param_shape)

            return derivate

        ref_param = np.zeros(distr_param_shape)
        ref_param[1] = np.ones(sample_dim)

        super().__init__(
            prob_map=prob_map,
            log_dens_der=log_dens_der,
            ref_param=ref_param,
            distr_param_shape=distr_param_shape,
            sample_shape=sample_shape,
        )
        self.sample_dim = sample_dim

    def kl(
        self,
        param1: ProbaParam,
        param0: ProbaParam,
        n_sample: int = 0,
    ) -> float:
        """
        Computes the Kullback Leibler divergence between two tensorized gaussian distributions
        defined by their distribution parameters.

        Args:
            distrib1, distrib0 are 2 distribution parameters
            n_sample is disregarded

        Output:
            kl(distrib1, distrib0)
        """
        dim = self.sample_dim

        means1, means0 = param1[0], param0[0]
        vars1, vars0 = param1[1] ** 2, param0[1] ** 2

        diff_mean = means1 - means0

        return 0.5 * (
            np.sum(np.log(vars0) - np.log(vars1))
            - dim
            + np.sum(vars1 / vars0)
            + np.sum((vars0 ** (-1)) * (diff_mean**2))
        )

    def grad_kl(self, param0: ProbaParam) -> Callable[[ProbaParam, int], ProbaParam]:

        mean0, vars0 = param0[0], param0[1] ** 2

        def fun(
            param1: np.ndarray,
            n_sample: int = 0,  # pylint: disable=W0613
        ) -> tuple[np.ndarray, float]:
            der = np.zeros(self.distr_param_shape)
            vars1 = param1[1] ** 2

            der[0] = (param1[0] - mean0) * (vars0 ** (-1))
            der[1] = param1[1] * (vars0 ** (-1) - vars1 ** (-1))

            return der, self.kl(param1, param0)

        return fun

    def grad_right_kl(
        self, param1: ProbaParam
    ) -> Callable[[ProbaParam, int], ProbaParam]:

        mean1, vars1 = param1[0], param1[1] ** 2

        def fun(
            param0: np.ndarray,
            n_sample: int = 0,  # pylint: disable=W0613
        ) -> tuple[np.ndarray, float]:
            der = np.zeros(self.distr_param_shape)
            vars0 = param0[1] ** 2
            diff_mean = param0[0] - mean1
            der[0] = diff_mean * (vars0 ** (-1))
            der[1] = (-(diff_mean**2) / vars0 + 1 - (vars1 / vars0)) / param0[1]

            return der, self.kl(param1, param0)

        return fun

    def to_param(self, g_distr: TensorizedGaussian) -> ProbaParam:
        """
        Transforms a Tensorized Gaussian back to a np.ndarray such that
        self.map(self.to_param(g_distr)) = g_distr
        """
        accu = np.zeros(self.distr_param_shape)
        accu[0] = g_distr.means
        accu[1] = g_distr.devs

        return accu


def tgauss_to_gauss_param(prob_param: ProbaParam) -> ProbaParam:
    """Convert a ProbaParam for TensorizedGaussianMap to a ProbaParam for ProbaMap resulting in
    the same gaussian distribution
    """
    prob_param = np.array(prob_param)
    distr_param_shape = prob_param.shape
    if len(distr_param_shape) != 2:
        raise ShapeError(
            "A ProbaParam for TensorizedGaussianMap should be 2 dimensional"
        )

    if distr_param_shape[0] != 2:
        raise ShapeError(
            "A ProbaParam for TensorizedGaussianMap should be shaped '(2, n)'"
        )

    accu = np.zeros((distr_param_shape[1] + 1, distr_param_shape[1]))
    accu[0] = prob_param[0]  # Passing means
    accu[1:] = np.diag(prob_param[1])  # Setting standard deviations

    return accu
