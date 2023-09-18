"""
Uncertainty quantification using residual bootstrapping
"""

from functools import partial
from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import ArrayLike

from ..misc import par_eval


def lin_bootstrap(
    res: np.ndarray,
    grad: np.ndarray,
    weights: Optional[np.ndarray] = None,
    n_boot: Optional[int] = None,
) -> np.ndarray:
    r"""
    Uncertainty quantification technique by bootstrapping residuals and using a linear
    approximation of the prediction model.

    For the statistical model
        $ Obs = f(\theta) + noise $
    the uncertainty is estimated by considering the approximate model
        $ Obs = \nabla f(\theta^*) (\theta - \theta^*) + noise $
    and bootstrapping the noise.

    This is computationnally more efficient than the full bootstrap procedure while gaining some
    robustness to non gaussian noise compared to Fisher's information matrix.

    Args:
        res: the residuals ($Obs - f(\theta)$)
        grad: the Jacobian $\nabla f(\theta^*)$
        weights: weights given to the residuals. Default is None (all weights = 1)
        n_boot: number of bootstrapped sample to generate. Default is None (N>1000, 10 * (dim ** 2)) with
            dim the dimension of the parameter fitted.

    Returns:
        the bootstrapped samples of parameters $\theta - \theta^*$ as numpy.ndarray.

    Input format:
        The residuals should be shaped (r1, ..., rn)
        The gradient should be shaped (p, r1, ..., rn) where p is the dimension of $\theta$
        The weights should be shaped (r1, ..., rn)

    About weights:
        To bootstrap, it is necessary to assume that noise is i.i.d. This can be partly relaxed by
        specifying weights. The assumption becomes that $noise * \sqrt{weights}$ are i.i.d. (this
        is coherent with the implementation of covariance estimation through fim).

    """

    # Set up None defaults
    if weights is None:
        weights = np.ones(res.shape)
    if n_boot is None:
        n_boot = max(1000, 10 * (grad.shape[0] ** 2))

    # Weigh residuals
    res_w = res * np.sqrt(weights)

    # Weigh gradient
    grad_w = grad * np.sqrt(weights)

    # Pre compute the linear solver
    n_dim = len(grad.shape)
    squared = np.tensordot(
        grad_w, grad_w, (tuple(range(1, n_dim)), tuple(range(1, n_dim)))
    )

    pseud_inv = np.linalg.inv(squared)
    solver = np.tensordot(pseud_inv, grad_w, (0, 0))
    # solver is a tensor of shape (p, r1, ..., rn)

    # Prepare bootstrapped residuals
    b_res = np.random.choice(
        res_w.flatten(), size=(n_boot,) + res_w.shape, replace=True
    )
    # b_res is a tensor of shape (n_boot, r1, ..., r_n)

    boot_sample = np.tensordot(
        b_res, solver, (tuple(range(1, n_dim)), tuple(range(1, n_dim)))
    )

    return boot_sample


def bootstrap(
    n_boot: int,
    calib: Callable[[np.ndarray], Any],
    obs: ArrayLike,
    opti_pred: ArrayLike,
    parallel: bool = True,
    **kwargs
) -> list:

    r"""
    Uncertainty quantification technique by bootstrapping residuals and recalibrating

    Args:
        n_boot: the number of bootstrapped samples to generate.
        calib: the calibration method. Takes an observation input and outputs a calibrated
            parameter
        opti_pred: predictions of the calibrated model. Used to compute the residuals.
        parallel: Specify whether the calibration procedures are parallelized

    Returns:
        the bootstrapped samples of parameters $\theta - \theta^*$ as a list

    Remark:
        Contrary to lin_bootstrap, the output is not a numpy.ndarray but a list.
    """
    opti_pred_arr = np.array(opti_pred).copy()
    res = np.array(obs) - opti_pred_arr

    res_shape = res.shape
    res_flat = res.flatten()

    calib_loc = partial(calib, **kwargs)

    predictions = [
        opti_pred_arr
        + np.random.choice(res_flat, size=len(res_flat)).reshape(res_shape)
        for _ in range(n_boot)
    ]

    parameters = par_eval(calib_loc, predictions, parallel=parallel)

    return parameters
