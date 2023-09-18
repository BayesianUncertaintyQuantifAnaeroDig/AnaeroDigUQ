"""
Probability objects module

Rules for maintenance

1. Which input functions should be vectorized?
Vectorization is assumed for transforms and internal functional attributes log_dens, log_dens_der
but it is not assumed for outside functions. That is to say, integrate and integrate_der expect
not vectorized function, all remaining functions are expected to be vectorized.

2. What do you mean by vectorization?
Vectorized is shape smart and is done on the left. I.E, if the base function expects inputs of shape
(s1, ..., sn) and outputs (o1, ...., om), then the vectorized function is meant to take inputs of shape
(shape_before, s1, ..., sn) to which it outputs (shape_before, o1, ..., om).

3. How can I vectorize my function?
Quite a lot of functions are "natively" vectorized, though perhaps not using the same convention (i.e.
witness lambda x : a @ x). The first thing should be to change the implementation if possible (e.g.
lambda x: np.tensordot(x, a, (-1, -1))). If that is not possible, then one can use the vectorize function
from aduq.misc 




For future considerations:
- Minimize function recursion added time when transforming.
- Potential speed gain when the transform functions are vectorized (i.e. for linear transforms,
    component wise transforms, etc...). Samples should be passed as sample_shape + (sample_size,)
    for these functions, contrary to current (sample_size,) + sample_shape. Use case: Uniform
    priors through Gaussians (spicy functions being vectorized).
- Notably, this question of vectorization is most important for log_dens evaluations, since many
    routines rely on list comprehension for evaluations of log densities at potentially many
    sample points. Moreover, most users should not consider implementing log_dens themselves but
    rather rely on already implemented distributions/distributions map and then use transforms.
"""

from ._types import ProbaParam, SamplePoint
from .exponential_family import ExponentialFamily
from .gauss import (
    FactCovGaussianMap,
    FixedCovGaussianMap,
    GaussHypercubeMap,
    Gaussian,
    GaussianMap,
    TensorizedGaussian,
    TensorizedGaussianMap,
)
from .proba import Proba, add, from_sample, mixture, tensorize
from .proba_map import ProbaMap, map_tensorize
