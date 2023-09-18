from typing import Optional

import numpy as np
from scipy.special import digamma, gamma

from .._helper import _get_pre_shape, _shape_info, prod
from .._types import ProbaParam, Samples
from ..proba_map import ProbaMap
from .gamma import Gamma


def _J_gamma_transform(mode: float, sigma: float):
    """Jacobian of the gamme transform as
    [[d k / d mode, dk / d sigma],
     [ d theta / dmode, d theta / dsigma]]
    """
    alpha = (mode / sigma) ** 2
    int_1 = np.sqrt(1 + 4 / alpha)

    k = 1 + alpha * (1 + int_1) / 2

    d_k_alpha = (1 + int_1) / 2 - (1 / (alpha * int_1))

    d_k_mode = 2 * alpha / mode * d_k_alpha
    d_k_sigma = -2 * alpha / sigma * d_k_alpha

    d_theta_mode = 1 / (k - 1) + d_k_mode * mode / ((k - 1) ** 2)
    d_theta_sigma = d_k_sigma * mode / ((k - 1) ** 2)

    return np.array([[d_k_mode, d_k_sigma], [d_theta_mode, d_theta_sigma]])


class GammaMap(ProbaMap):
    def __init__(
        self,
        sample_dim: Optional[int] = None,
        sample_shape: Optional[tuple[int]] = None,
    ):
        sample_dim, sample_shape = _shape_info(sample_dim, sample_shape)
        distr_param_shape = (2, sample_dim)

        def prob_map(proba_param: ProbaParam) -> Gamma:
            proba_param = np.array(proba_param)
            par = proba_param.reshape((2,) + sample_shape)
            return Gamma(modes=np.abs(par[0]), sigmas=np.abs(par[1]))

        def log_dens_der(distrib_param: ProbaParam):
            distrib_param = np.array(distrib_param)
            distr = prob_map(distrib_param)

            ks, thetas = distr.ks, distr.thetas
            modes, sigmas = distr.modes, distr.sigmas

            _mats = [
                _J_gamma_transform(mode, sigma) for mode, sigma in zip(modes, sigmas)
            ]
            c_d_log_ks = -np.log(thetas) - digamma(ks)
            c_d_log_thetas = -ks / thetas

            c_d_log_ms = [
                mat @ np.array([d_k, d_theta])
                for mat, d_k, d_theta in zip(_mats, c_d_log_ks, c_d_log_thetas)
            ]
            accu = np.array(c_d_log_ms).T

            def der(samples: Samples) -> np.ndarray:
                pre_shape = _get_pre_shape(samples, sample_shape)
                loc_acc = np.full((prod(pre_shape),) + distr_param_shape, accu)

                d_log_k = np.log(samples)
                d_log_theta = samples / (thetas**2)

                for i in sample_dim:
                    loc_acc[:, i] += _mats[i] @ [d_log_k[i], d_log_theta[i]]
                    loc_acc[:, i] = loc_acc[:, i] * np.sign([modes[i], sigmas[i]])
                return loc_acc.reshape(pre_shape + distr_param_shape)

            return der

        super().__init__(
            prob_map=prob_map,
            log_dens_der=log_dens_der,
            ref_param=np.ones(distr_param_shape),
            distr_param_shape=distr_param_shape,
            sample_shape=sample_shape,
        )

    def kl(
        self,
        param1: ProbaParam,
        param0: ProbaParam,
        n_sample: int = 0,
    ):
        distr1, distr0 = self.map(param1), self.map(param0)
        ks1, thetas1 = distr1.ks, distr1.thetas
        ks0, thetas0 = distr0.ks, distr0.thetas

        digammas = digamma(ks1)
        out = (
            (ks1 - 1) * digammas
            - np.log(thetas1)
            - ks1
            - np.log(gamma(ks1))
            + np.log(gamma(ks0))
            + ks0 * np.log(thetas0)
            - (ks0 - 1) * (digammas + np.log(thetas1))
            + thetas1 * ks1 / ks0
        )

        return out
