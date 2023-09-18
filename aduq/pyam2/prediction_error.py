"""
Define prediction error as -mostly- a RMSE betwwen the prediction and observations

Main functions:

- am2_error, compute the error from predictions
- score_param, compute the error from a parameter
- score_free_param, compute the error from a "free" parameter (parametrization without constraint
used during optimisation)
"""

import warnings
from typing import Optional

import numpy as np

from ._typing import DigesterFeed, DigesterParameter, DigesterState, DigesterStates
from .interp_param import interp_param
from .IO import err_pred_ind
from .run_am2 import run_am2


class AM2Failure(Warning):
    """Warning when AM2 raised an exception which is later caught"""


class TimeoutWarning(Warning):
    """Warning for excessive duration time (process is ended)"""


class UnknownKwargs(Warning):
    """Warning when unknown kwargs are passed to a function"""


def am2_err(
    pred: DigesterStates,
    obs: DigesterStates,
    normalisation: Optional[list] = None,
    eps: float = 10 ** (-8),
    **kwargs,  # pylint: disable=W0613
) -> float:
    r"""
    Compute the error as RMSE of log.
    Predictions used to compute the error are

    Args:
        pred, prediction from AM2
        obs, digester observation,
        normalisation, renormalisation for each type of prediction
        eps, value to avoid numerical errors

    kwargs:
        All further kwargs are disregarded. Throws a warning if any (but does not fail).

    Output:
        prediction error as
            $$\sqrt( (\sum_i \omega_i \sum_t \log( (pred_{i,t} + eps)
            / (obs_{i,t} + eps)) **2) / \sum_i t_i \omega_i )$$
        with t_i the number of non nan data for prediction type i,
        $\omega_i$ a renormalisation factor (by default 1, else from normalisation).
        NaNs are ignored during computation.

    Future:
        Remove conditions on shape and change to alignment of time information.
    """

    if len(kwargs) > 0:
        warnings.warn(
            f"Arguments {list(kwargs.keys())} are not used", category=UnknownKwargs
        )
    # Check shape
    if pred.shape != obs.shape:
        raise Exception(
            "Error can only be computed between 'pred' and 'obs' of same shape."
        )

    # Subsetting to features used for error computations
    pred_f = np.asarray(pred[:, err_pred_ind])
    obs_f = np.asarray(obs[:, err_pred_ind])
    # Computing log-residuals
    res = np.log((pred_f + eps) / (obs_f + eps))

    # Case where no renormalisation
    if normalisation is None:
        # Using nanmean here. Better practice would be, if any value in pred is nan, should be inf,
        # while nan values in obs should be disregarded.
        return np.sqrt(np.nanmean(res**2))

    normalisation = np.array(normalisation)
    count_non_nan = np.array([np.sum(~np.isna(res[:, i])) for i in range(res.shape[1])])

    err_per_pred = np.array([np.nansum(res[:, i] ** 2) for i in range(res.shape[1])])
    err_tot = np.sum(normalisation * err_per_pred)
    err = err_tot / np.sum(normalisation * count_non_nan)

    return np.sqrt(err)


def score_param(
    param: DigesterParameter,
    obs: DigesterStates,
    influent_state: DigesterFeed,
    initial_state: DigesterState,
    max_score: float = 20.0,
    **kwargs,
) -> float:
    """
    Compute the score of a parameter.

    Args:
        param: a digester parameter
        obs: digester states observed
        influent_state, initial_state, digester_info: see run_adm1
        max_score: Score used when ADM1 computations fail
    kwargs:
        solver_method, log_path, stop_on_err and scipy solver **kwargs are used by run_adm1
        normalisation, eps are used by am2_err

    Code maintenance:
        Use of filterwarning. What must be filtered are numerical prediction failures.
    """

    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=RuntimeWarning)
        warnings.filterwarnings("error", category=UserWarning)
        try:
            pred = run_am2(
                param,
                influent_state=influent_state,
                initial_state=initial_state,
                **kwargs,
            )
            score = am2_err(pred, obs, **kwargs)
        except (RuntimeWarning, UserWarning) as exc:
            warnings.warn(
                f"Could not compute error for parameter:\n{param}\n\n{exc}",
                category=AM2Failure,
            )
            score = max_score
    return score


score_free_param = interp_param(score_param)
