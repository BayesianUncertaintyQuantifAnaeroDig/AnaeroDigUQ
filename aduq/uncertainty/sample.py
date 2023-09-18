""" Uncertainty quantification from a sample """
import warnings
from math import ceil
from typing import Callable, Optional

import numpy as np
from scipy.stats import binom
from sklearn.ensemble import IsolationForest


def sample_pval(param, sample, sample_weight=None, train_ratio=0.3, **kwargs):
    """Compute approximate p-value of whether a parameter is drawn for the same distirbution
    as a sample.

    This technique is based on IsolationForest function from scikit-learn package. An anomaly
    function is trained on part of the sample, calibrated using the remaining part

    Args:
        param: a point
        sample: a presumedly i.i.d. sample from an unknown distribution
        sample_weight: weights given to each sample. Optional (default None implies equal weights)
        train_ratio: fraction of the sample used to train the sample
    Further kwargs are passed to IsolationForest constructor.
    """
    n_data = len(sample)
    n_train = int(n_data * train_ratio)

    sample = np.array(sample)

    sample = sample.reshape((len(sample), round(np.prod(sample.shape[1:]))))
    param = np.array(param).flatten()

    indices = np.random.permutation(len(sample))

    train_data, calib_data = sample[indices[:n_train]], sample[indices[n_train:]]

    if sample_weight is None:
        train_weight, calib_weight = None, None
    else:
        train_weight, calib_weight = (
            sample_weight[indices[:n_train]],
            sample_weight[indices[n_train:]],
        )
        train_weight, calib_weight = train_weight / np.sum(
            train_weight
        ), calib_weight / np.sum(calib_weight)

    forest = IsolationForest(**kwargs)
    forest.fit(train_data, sample_weight=train_weight)

    scores_calib = forest.score_samples(calib_data)
    score_param = forest.score_samples(np.array([param]))[0]

    if calib_weight is None:
        pval = np.mean(score_param > scores_calib)
    else:
        pval = np.sum(calib_weight[score_param > scores_calib])

    return pval


def dichoto(
    fun: Callable[[float], float],
    y: float,
    x_min: float = 0,
    x_max: Optional[float] = None,
    increasing: bool = False,
    y_pres: Optional[float] = None,
    x_pres: Optional[float] = None,
    m_max: int = 1000,
):
    """
    Solves fun(x) = y for monotonous functions
    """

    # Convert decreasing to increasing if needed
    mult = (-1) ** (increasing + 1)
    y = y * mult

    compt = 0

    if x_max is None:

        if x_min < 0:
            compt += 1
            if mult * fun(0) > y:
                x_max = 0
                x_max_routine = False
            else:
                # 0 is not large enough. Start guess with - x_min
                x_max = -x_min
                x_min = 0
                x_max_routine = True

        elif x_min == 0:
            x_max = 1.0
            x_max_routine = True

        else:
            x_max = 2 * x_min
            x_max_routine = True

        if x_max_routine:
            while (mult * fun(x_max) < y) & (compt < m_max):
                x_min = x_max
                x_max = 2 * x_max
                compt += 1
            if compt >= m_max:
                raise Exception(
                    f"Could not find an interval containing {y}. Last couple: {(x_min, x_max)}"
                )
    else:
        # Check that interval contains y
        f_min = mult * fun(x_min)
        f_max = mult * fun(x_max)
        if (y < f_min) or (y > f_max):
            raise Exception(
                f"Interval (f(x_min), f(x_max)) does not contain {mult * y}"
            )

    max_pres = 2 ** (-(m_max - compt))

    if x_pres is None:
        n_iter = m_max - compt
        x_pres = 10 ** (-4) * (x_max - x_min)
    elif max_pres > x_pres:
        warnings.warn("The required precision in x could not be achieved")
        return (x_min, x_max)
    else:
        n_iter = ceil(
            np.log2((x_max - x_min) / x_pres)
        )  # Number of iterations for precision

    for _ in range(n_iter):
        x_new = (x_max + x_min) / 2
        f_new = mult * fun(x_new)
        if f_new >= y:
            x_max = x_new
        else:
            x_min = x_new
        compt += 1

    # Now enforce precision on y.
    if y_pres is None:
        return (x_min, x_max)
    else:
        err_min = y - mult * fun(x_min)
        err_max = mult * fun(x_max) - y

        while (compt < m_max) & (max(err_min, err_max) > y_pres):

            x_new = (x_max + x_min) / 2
            f_new = mult * fun(x_new)
            if f_new >= y:
                x_max = x_new
                err_max = f_new - y
            else:
                x_min = x_new
                err_min = f_new

            compt += 1
        if compt >= m_max:
            warnings.warn("The required precision in x could not be achieved")
        return (x_min, x_max)


def binom_bound(
    k: int,
    n: int,
    alpha: float = 0.05,
    p_pres: Optional[float] = None,
    y_pres: Optional[float] = None,
    m_max: int = 1000,
):
    """Confidence upper bound of the mean of a Bernoulli given a sample of mean k/n

    The bound is true with probability at least 1 - alpha
    """

    # Prepare constant
    if k == 0:
        ratio = np.log(alpha) / n
        if (-ratio) < 10 ** (-8):
            return -ratio
        return 1 - np.exp(ratio)

    if p_pres is None:
        p_pres = min((max(k, 1) / n) ** 2, 0.01)

    if y_pres is None:
        y_pres = alpha * 10 ** (-4)

    def fun(p):
        return binom(n, p).cdf(k)

    pmax = 1.0
    pmin = k / n

    pmin, pmax = dichoto(
        fun=fun,
        y=alpha,
        x_min=pmin,
        x_max=pmax,
        increasing=False,
        y_pres=y_pres,
        x_pres=p_pres,
        m_max=m_max,
    )
    return pmax  # Conservative output


def upper_bound_sample_pval(param, sample, train_ratio=0.3, conf_lev=0.95, **kwargs):
    """Compute confidence upper bound of p-value of whether a parameter is drawn for the same
    distribution as a sample.

    This technique is based on IsolationForest function from scikit-learn package. An anomaly
    function is trained on part of the sample, calibrated using the remaining part. A test bound is
    used to compute the confidence upper bound of the p-value.

    Args:
        param: a point
        sample: a presumedly i.i.d. sample from an unknown distribution
        sample_weight: weights given to each sample. Optional (default None implies equal weights)
        train_ratio: fraction of the sample used to train the sample
        conf_lev: required confidence level for the p-value upper bound
    Further kwargs are passed to IsolationForest constructor.
    """
    n = len(sample)
    n_calib = n - int(n * train_ratio)

    k = n_calib * sample_pval(
        param=param, sample=sample, train_ratio=train_ratio, **kwargs
    )

    return binom_bound(k, n_calib, alpha=1 - conf_lev)
