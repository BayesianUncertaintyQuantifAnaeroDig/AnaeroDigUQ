""" Exponential distributions """

from typing import Callable, Iterable

import numpy as np

from .._helper import prod
from .._types import ProbaParam, SamplePoint, Samples
from .exponential_family import ExponentialFamily


def __T(x: Samples) -> np.ndarray:
    return x


def __g(par: ProbaParam) -> float:
    return -np.log(par[0])


def __der_g(par: ProbaParam) -> ProbaParam:
    return -1 / par


def __der_der_g(par: ProbaParam) -> np.ndarray:
    return (1 / (par**2)).reshape((1, 1))


def __gen(par: ProbaParam) -> Callable[[int], Iterable[SamplePoint]]:
    def fun(n):
        return np.random.exponential(1 / par[0], (n, 1))

    return fun


def __h(x: Samples) -> np.ndarray:
    pre_shape = x.shape[:-1]
    x = x.flatten()
    out = np.zeros(prod(pre_shape))
    out[~np.apply_along_axis(lambda x: x[0] > 0.0, -1, x)] = -np.inf
    return out.reshape(pre_shape)


Exponential = ExponentialFamily(
    gen=__gen,
    T=__T,
    g=__g,
    der_g=__der_g,
    der_der_g=__der_der_g,
    h=None,
    distr_param_shape=(1,),
    sample_shape=(1,),
    ref_param=np.array([1]),
    np_out=True,
)
