r"""

Beta distributions

Density is
    $$ \exp(
        (\alpha - 1)\log(x) + (\beta - 1)\log(1-x)
        + \log(\Gamma(\alpha))) + \log(\Gamma(\beta))
        - \log(\Gamma(\alpha + \beta))
        )$$

for x in [0,1], 0 if not.

"""

from typing import Callable, Iterable

import numpy as np
from scipy.special import digamma, gamma, polygamma

from .._helper import prod
from .._types import ProbaParam, SamplePoint, Samples
from .exponential_family import ExponentialFamily


def __T(x: Samples) -> ProbaParam:
    pre_shape = x.shape
    out = np.zeros((prod(pre_shape), 2))
    x = x.flatten()
    good_index = (x > 0) and (x < 1)

    out[good_index] = np.array([np.log(x[good_index]), np.log(1 - x[good_index])]).T
    return out.reshape(pre_shape + (2,))


def __g(par: ProbaParam) -> float:
    return np.sum(np.log(gamma(par))) - np.log(gamma(np.sum(par)))


def __der_g(par: ProbaParam) -> ProbaParam:
    return digamma(par) - digamma(np.sum(par))


def __der_der_g(par: ProbaParam) -> np.ndarray:
    return np.diag(polygamma(1, par)) - polygamma(1, np.sum(par))


def __h(x: Samples) -> np.ndarray:
    pre_shape = x.shape[:-1]
    x = x.flatten()
    out = np.zeros(prod(pre_shape))
    out[~np.apply_along_axis(lambda x: (x[0] < 1.0) and (x[0] >= 0.0), -1, x)] = -np.inf
    return out.reshape(pre_shape)


def __gen(par: ProbaParam) -> Callable[[int], Iterable[SamplePoint]]:
    def fun(n: int) -> Iterable[SamplePoint]:
        return np.random.beta(par[0], par[1], size=n)

    return fun


Beta = ExponentialFamily(
    gen=__gen,
    T=__T,
    g=__g,
    der_g=__der_g,
    der_der_g=__der_der_g,
    h=__h,
    distr_param_shape=(2,),
    sample_shape=(1,),
    ref_param=np.array([1, 1]),
    np_out=True,
)
