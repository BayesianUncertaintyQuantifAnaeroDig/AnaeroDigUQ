import warnings
from functools import partial
from typing import Optional

import numpy as np

from ...misc import blab, par_eval, safe_call, timedrun
from ..IO import (
    DigesterFeed,
    DigesterInformation,
    DigesterParameter,
    DigesterState,
    DigesterStates,
)
from ..run_adm1 import run_adm1


def weighted_quantile(
    values, quantiles, sample_weight=None, values_sorted=False, old_style=False
):
    """Very close to numpy.percentile, but supports weights.
    Note: quantiles should be in [0, 1]!
    Adapted from https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(
        quantiles <= 1
    ), "quantiles should be in [0, 1]"

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


def prediction_uncertainty_from_output(
    predictions: list[DigesterStates],
    weights: Optional[list[float]] = None,
    min_max: bool = False,
    quantiles: Optional[list[float]] = None,
) -> np.ndarray:
    """Quantify uncertainty on predictions from a list of predictions
    Args:
        predictions: a list of predictions representing the uncertainty of the predictions
        weights: weights given to each element in the list of predictions
        min_max: Should the uncertainty be given as ranges between minimum and maximum values
            obtained (this is the case for Beale's method)?
        quantiles: Quantiles of the predictions desired.

    Outputs a 3 dimensional np.ndarray. Subsetting on the first element gives a DigesterStates
        like object.
         - If min_max, shape is (2, n_days, n_obs), the first one is the min, the second one is
            the max.
         - If quantiles, shape is (len(quantiles), n_days, n_obs), and the output follows the order
            of quantiles.
    """

    if quantiles is None:
        quantiles = [0.25, 0.75]

    # Converting to np.ndarray for easier manipulation
    preds = np.array(predictions)
    if min_max:

        def min_max_func(vals):
            return np.array([vals.min(), vals.max()])

        return np.apply_along_axis(min_max_func, 0, preds)

    if weights is None:
        weights = np.full(len(preds), 1 / len(preds))
    else:
        weights = np.array(weights)

    # Remove predictions with too little mass (at most 0.005 mass is removed + checks so that
    # it does not affect quantiles precisions too much)
    weight_threshold = (
        10 ** (-2) * min(np.min(quantiles), 1 - np.max(quantiles)) / len(weights)
    )
    index = weights > weight_threshold

    preds = preds[index]
    weights = weights[index]
    weights = weights / np.sum(weights)  # Renormalise weights to 1.0

    w_quantile = partial(weighted_quantile, sample_weight=weights, quantiles=quantiles)

    return np.apply_along_axis(w_quantile, 0, np.array(predictions))


def prediction_uncertainty_from_param(
    parameters: list[DigesterParameter],
    influent_state: DigesterFeed,
    initial_state: DigesterState,
    digester_info: DigesterInformation,
    weights: Optional[list[float]] = None,
    min_max: bool = False,
    quantiles: Optional[list[float]] = None,
    max_time: int = 300,
    parallel: bool = True,
    silent: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Uncertainty in prediction space from weighted parameters.

    Args:
        parameters, a list of parameters considered to be a sample
        influent_state, initial_state, digester_info, and **kwargs are passed to run_adm1 routine
        weights, a list of weights for each parameter.
        min_max, state whether the uncertainty quantification requires taking the min and max
            across predictions from sampled parameters (this is the case for Beale's method)
            Default is False, so quantiles are used instead.
        quantiles, a list specifying which quantiles of the output should be considered.
        max_time, the maximum duration (in seconds) of a call to run_adm1 routine before it is
            stopped. Unsuccessfull calls are afterwards disregarded. Default is 5 minutes.
        parallel, should the computations be parallelized? Default is True

    Output:
        A list of same size as quantiles, giving the multidimensional time series associated with
        each quantile required.

    Future:
        So far, the weight removal mechanism is performed during prediction_uncertainty_from_output
        step. This is suboptimal, since costly run_adm1 is evaluated on parameters afterwards
        discarded.
    """

    blab(silent, "Prediction uncertainty from sample of parameters routine")

    if quantiles is None:
        quantiles = [0.25, 0.75]

    if weights is None:
        weights = np.ones(len(parameters)) / len(parameters)

    adm1_loc = partial(
        run_adm1,
        influent_state=influent_state,
        initial_state=initial_state,
        digester_info=digester_info,
        **kwargs,
    )

    safe_adm1 = safe_call(timedrun(max_time)(adm1_loc))

    blab(silent, "ADM1 computations started.")
    predictions = par_eval(safe_adm1, parameters, parallel=parallel)
    blab(silent, "ADM1 computations over.")

    # Remove unusable predictions
    bad_indexes = [x is None for x in predictions]

    if any(bad_indexes):
        tot_weight_rm = np.sum(np.array(weights)[bad_indexes])
        warnings.warn(f"A mass of {tot_weight_rm} parameters could not be assessed")

        predictions = [x for x in predictions if x is not None]
        weights = np.array(weights)[~np.array(bad_indexes)]
        weights = weights / np.sum(weights)

    predictions = np.array(predictions)

    pred_quantiles = prediction_uncertainty_from_output(
        predictions=predictions, weights=weights, min_max=min_max, quantiles=quantiles
    )
    return {
        "sim": predictions,
        "weights": weights,
        "pred_quantiles": pred_quantiles,
        "quantiles": quantiles,
    }
