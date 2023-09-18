"""
Gamma Distributions
"""

from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import loggamma  # pylint: disable=E0611

from .._helper import _get_pre_shape, prod
from .._types import SamplePoint, Samples
from ..proba import Proba


def _gamma_transform(
    modes: ArrayLike, sigmas: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Transformation to change mode/sigma distribution information
    into parameters specific to gamma distributions (k, theta)"""
    modes, sigmas = np.array(modes), np.array(sigmas)

    alpha = (modes / sigmas) ** 2
    k = 1 + alpha * (1 + np.sqrt(1 + 4 / alpha)) / 2
    theta = modes / (k - 1)
    return (k, theta)


class Gamma(Proba):
    """
    Gamma distributions
    Inherited from Proba class. Constructed from modes and standard deviations.
    """

    def __init__(self, modes: ArrayLike, sigmas: ArrayLike):

        if isinstance(modes, float) & isinstance(sigmas, float):
            modes, sigmas = np.array([modes]), np.array([sigmas])
        else:
            modes, sigmas = np.array(modes), np.array(sigmas)

        sample_shape = modes.shape
        modes, sigmas = modes.flatten(), sigmas.flatten()

        n_dim = len(modes)
        if len(sigmas) != n_dim:
            raise Exception("Length mismatch between modes and sigmas")
        ks, thetas = _gamma_transform(modes, sigmas)

        const_log_dens = -np.sum((ks * np.log(thetas) + loggamma(ks)))

        def gen(n: int) -> Iterable[SamplePoint]:
            accu = np.zeros((n, n_dim))
            for i, (k, theta) in enumerate(zip(ks, thetas)):
                accu[:, i] = np.random.gamma(shape=k, scale=theta, size=n)
            return accu.reshape((n,) + sample_shape)

        def log_dens(samples: Samples) -> np.ndarray:
            samples = np.array(samples)
            pre_shape = _get_pre_shape(samples, sample_shape)
            samples = samples.reshape(pre_shape + (prod(sample_shape),))

            return const_log_dens + (np.log(samples) * (ks - 1) - samples / thetas).sum(
                -1
            )

        super().__init__(
            gen=gen, log_dens=log_dens, sample_shape=sample_shape, np_out=True
        )
        self.modes = modes
        self.sigmas = sigmas
        self.ks = ks
        self.thetas = thetas
