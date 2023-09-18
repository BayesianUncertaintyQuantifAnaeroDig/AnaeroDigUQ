"""Error computation for ADM1 module

Main functions:
    - adm1_err: compute error from predictions and observations
    - score_param: compute error from parameter, observations and further digester information
    - score_free_param: equivalent of score_param for FreeDigesterParameter space

score_param and score_free_param are used during calibration and UQ procedure.
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from ..misc import ShapeError
from .IO import (
    DigesterFeed,
    DigesterInformation,
    DigesterParameter,
    DigesterState,
    DigesterStates,
    interp_param,
    pred_names_to_index,
    small_predictions_col,
)
from .run_adm1 import run_adm1


class ADM1Failure(Warning):
    """Warning class when ADM1 computations failed"""


def soft_plus(x: float, max_val: float = 3.0, elbow: float = 1.0) -> float:
    """
    Variation on the softplus function used for capping.

    Smoothed version of x -> np.max(max_val, np.abs(x)).
    args:
        x, a float to be smoothly capped (if np.ndarray, applied element wise)
        max_val, the maximum value returned as x -> infty
        elbow, a control on the smoothness

    """
    prop = (1 + np.exp(-elbow * max_val)) / elbow
    C = np.log(1 + np.exp(elbow * max_val))
    return prop * (C - np.log(1 + np.exp(elbow * (max_val - np.abs(x)))))


default_pred_names = small_predictions_col[1:]


def adm1_err(
    pred: DigesterStates,
    obs: DigesterStates,
    pred_names: list[str] = None,
    normalisation: Optional[pd.Series] = None,
    eps: float = 10 ** (-8),
    max_score: float = 3.0,
    elbow: float = 2.0,
    **kwargs,  # pylint: disable=W0613
) -> float:
    """
    Compute the error as pseudo Root mean square of log residuals

    Args:
        pred prediction from ADM1
        obs: digester observation
        pred_names: List of names of types of predictions to be used. By default all names in
            small predictions except time
        normalisation: should some normalisation (on types of prediction) be used? Stored as a
            pd.Series specifying the weights to be applied to each type of predictions.
            Prediction types not in the index are removed.
        eps: soft threshold for small values (i.e. if both pred and obs are << eps, the error
            contribution is close to 0)
        max_score: soft threshold for large error contribution
        elbow: smoothness of thresholding for large error contribution

    Output:
        prediction error as
            sqrt( (sum_i omega_i sum_t log(pred_{i,t}/obs_{i,t}) **2) / sum_i t_i omega_i )
        with t_i the number of non nan data for prediction type i, omega_i a renormalisation factor
        (by default 1). nan are ignored. The sum is performed only on pred_names
    """

    if pred_names is None:
        # Double check python scoping rules. default_pred_names used should be the one in this
        # module, not a potential user defined variable of the same name
        pred_names = default_pred_names

    # Check shape
    if pred.shape != obs.shape:
        raise ShapeError("Error can only be computed for predictions of similar shape.")

    # Case with no normalisation (default case)
    if normalisation is None:
        # Rmv excess info
        col_ind = pred_names_to_index(pred_names)
        pred_f = np.array(pred[:, col_ind])
        obs_f = np.array(obs[:, col_ind])

        res = np.log((pred_f + eps) / (obs_f + eps))
        # Pass residuals through smooth capping
        corr_res = soft_plus(res, max_val=max_score, elbow=elbow)

        return np.sqrt(np.nanmean(corr_res**2))

    # Rmv excess info
    columns_to_use = list(set(normalisation.index).intersection(pred_names))
    col_ind = pred_names_to_index(columns_to_use)

    normalisation = np.array(normalisation[columns_to_use])

    pred_f = np.array(pred[:, col_ind])
    obs_f = np.array(obs[:, col_ind])

    res = np.log((pred_f + eps) / (obs_f + eps))
    corr_res = soft_plus(res, max_val=max_score, elbow=elbow)

    count_non_nan = np.array(
        [np.sum(~np.isna(corr_res[:, i])) for i in range(res.shape[1])]
    )

    err_per_pred = np.array(
        [np.nansum(corr_res[:, i] ** 2) for i in range(res.shape[1])]
    )
    err_tot = np.sum(normalisation * err_per_pred)
    err = err_tot / np.sum(normalisation * count_non_nan)
    return np.sqrt(err)


def score_param(
    param: DigesterParameter,
    obs: DigesterStates,
    influent_state: DigesterFeed,
    initial_state: DigesterState,
    digester_info: DigesterInformation,
    pred_names: list[str] = None,
    normalisation: Optional[list] = None,
    eps: float = 10 ** (-8),
    max_score: float = 3.0,
    elbow: float = 1.0,
    silent: bool = False,  # pylint: disable=W0613 # prevent silent from being sent to scipy.linalg.solve
    **kwargs,
) -> float:
    """
    Score a parameter by computing its prediction and computing prediction error
    """
    try:
        preds = run_adm1(
            param,
            influent_state=influent_state,
            initial_state=initial_state,
            digester_info=digester_info,
            **kwargs,
        )
        return adm1_err(
            preds,
            obs,
            pred_names=pred_names,
            normalisation=normalisation,
            eps=eps,
            max_score=max_score,
            elbow=elbow,
        )
    except (
        RuntimeWarning,
        UserWarning,
    ) as exc:  # These are filtered as exceptions during run_adm1
        warnings.warn(
            f"Could not compute error for parameter:\n{param}\n\n{exc}",
            category=ADM1Failure,
        )
        return max_score


# Define the score function for FreeDigesterParameter
score_free_param = interp_param(score_param)
