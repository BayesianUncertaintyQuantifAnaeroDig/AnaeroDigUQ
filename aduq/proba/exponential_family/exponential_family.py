r"""
Class for Exponential family of probability distributions.

Exponential families, using the natural parametrisation, have densities
$$ f_\theta(x) = \exp(\theta \cdot T(x) - g(\theta) + h(x)) $$
with respect to a common distribution.

The Kullback--Leibler divergence has a closed form expression which amounts to a Bregman divergence
$$ KL(f_a, f_b) = g(b) - g(a) - (b - a) . nabla g(a).$$

This allows for easy differentiation, provided the Hessian of $g$ is known.

Reference:
    https://www.lix.polytechnique.fr/~nielsen/EntropyEF-ICIP2010.pdf

Note:
    Exponential family can be used to obtain another parametrizsation of Gaussian distributions as
        well as Gamma distributions. These specific implementations are expected to be somewhat
        more efficient though.
    Tensorization of Exponential families are also exponential families. While this information is
        lost, the map_tensorize function is coded in such a way as to ensure efficiency when
        computing kl, grad_kl, grad_right_kl methods.
"""
import warnings
from typing import Callable, Iterable, Optional

import numpy as np

from .._types import ProbaParam, SamplePoint, Samples
from ..proba import Proba
from ..proba_map import ProbaMap
from ..warnings import MissingShape


class ExponentialFamily(ProbaMap):
    r"""
    Subclass of ProbaMap for Exponential families.

    Exponential families have densities of form
        $$f_\†heta(x) = \exp(\theta \cdot T(x) - g(\theta) + h(x))$$

    (h can be omitted since it can be hidden in the reference measure).

    Many families of distributions are exponential families (gaussians, gamma, etc).
    """

    def __init__(  # pylint: disable=R0913
        self,
        gen: Callable[[ProbaParam], Callable[[int], Iterable[SamplePoint]]],
        T: Callable[[Samples], np.ndarray],
        g: Callable[[ProbaParam], float],
        der_g: Callable[[ProbaParam], ProbaParam],
        der_der_g: Optional[Callable[[ProbaParam], np.ndarray]] = None,
        h: Optional[Callable[[Samples], np.ndarray]] = None,
        distr_param_shape: Optional[tuple] = None,
        sample_shape: Optional[tuple] = None,
        ref_param: Optional[ProbaParam] = None,
        np_out: Optional[bool] = None,
    ):
        r"""
        Probaution map for an exponential family defined through its natural parameters

            $f_{\theta}(x) = \exp(\theta. T(x) - g(\theta) + h(x))$

        where f is the density.

        Natural parametrisation is required to efficiently compute KL. For change of parametrisation,
        use reparametrize which maintains efficient computation of KL and its gradient.
        """
        normed_log_dens = h is not None

        if (ref_param is None) & (distr_param_shape is None):
            warnings.warn(
                "No shape information on expected distribution parameters",
                category=MissingShape,
            )

        if distr_param_shape is None:
            distr_param_shape = np.array(ref_param).shape

        # Define dimensions on which to sum for log_dens function
        dims_log_dens_help = tuple(-i - 1 for i in range(len(distr_param_shape)))

        def prob_map(distr_param: ProbaParam) -> Proba:
            """Transforms a distribution parameter into a distribution (Proba object)"""
            loc_gen = gen(distr_param)

            g_loc = g(distr_param)
            if normed_log_dens:

                def log_dens(samples: Samples) -> np.ndarray:
                    # Samples should be of shape (pre_shape, sample_shape)
                    # T(samples) is of shape (pre_shape, distr_param_shape)
                    # h(samples) of shape (pre_shape,)

                    return (
                        (distr_param * T(samples)).sum(axis=dims_log_dens_help)
                        - g_loc
                        + h(samples)
                    )

            else:

                def log_dens(samples: Iterable[SamplePoint]) -> np.ndarray:
                    return distr_param * T(samples) - g_loc

            return Proba(
                gen=loc_gen, log_dens=log_dens, sample_shape=sample_shape, np_out=np_out
            )

        def log_dens_der(distr_param):
            g_der_loc = der_g(distr_param)

            def der(samples: Samples):
                return T(samples) - g_der_loc

            return der

        super().__init__(
            prob_map=prob_map,
            log_dens_der=log_dens_der,
            ref_param=ref_param,
            distr_param_shape=distr_param_shape,
            sample_shape=sample_shape,
        )

        self.g = g
        self.der_g = der_g
        self.H_g = der_der_g
        self.T = T
        self.h = h

    def kl(
        self,
        param1: ProbaParam,
        param0: ProbaParam,
        n_sample: Optional[int] = None,
    ):
        """
        Computes the Kullback Leibler divergence between two distributions
        defined by their prior parameters.

        Args:
            param1, param0 are 2 prior parameters
            n_sample, parallle: Disregarded

        Output:
            KL(distrib1, distrib0) computed through
                g(param0) - g(param1) - (param0 - param1) . nabla g(param1)

        Reference:
            https://www.lix.polytechnique.fr/~nielsen/EntropyEF-ICIP2010.pdf
        """

        par1, par0 = np.array(param1), np.array(param0)
        return (
            self.g(param0) - self.g(param1) - np.sum((par0 - par1) * self.der_g(param1))
        )

    def grad_right_kl(
        self, param1: ProbaParam
    ) -> Callable[[ProbaParam, int], tuple[ProbaParam, float]]:
        der_g1 = self.der_g(param1)

        def der(param0, n_sample: int = 0):  # pylint: disable=W0613
            return self.der_g(param0) - der_g1, self.kl(param1, param0)

        return der

    def grad_kl(
        self, param0: ProbaParam
    ) -> Callable[[ProbaParam, int], tuple[ProbaParam, float]]:
        """
        Approximates the gradient of the Kullback Leibler divergence between two distributions
        defined by their distribution parameters, with respect to the first distribution
        (nabla_{param1} KL(param1, param0))

        Args:
            param0 is a distribution parameter

        Output:
            If the hessian of the renormalisation is known, then this is used to compute the gradient.
            Else falls back to standard computations.

        Reference:
            Starting back from the formula for KLs of exponential families,
                KL(distrib1, distrib0) =
                    g(param0) - g(param1) - (param0 - param1) . nabla g(param1)
            it follows that the gradient of the kl wrt to param1 is
                Hessian(g)(param1) (param1 - param0)
        """
        if self.H_g is None:
            return ProbaMap.grad_kl(self, param0=param0)

        indices = list(range(len(self.distr_param_shape)))

        def der(param1: ProbaParam, n_sample: int = 0):  # pylint: disable=W0613
            par1, par0 = np.array(param1), np.array(param0)
            return np.tensordot(
                self.H_g(param1),  # (distr_param_shape, distr_param_shape)
                (par1 - par0),  # (distr_param_shape)
                [indices, indices],
            ), self.kl(param1, param0)

        return der
