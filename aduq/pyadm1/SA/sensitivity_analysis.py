"""
Sensitivity Analysis for ADM1.

Main functions:
    local_sensitivity
    global_sensitivity

local_sensitivity returns a pd.Series object, describing the sensitivity of outputs to the
    parameter (low values -> low impact).

global_sensitivity returns a pd.DataFrame object with two columns, "mean" and "sd". "mean" gives
    the mean sensitivity for each parameter, while "sd" gives the standard deviation of the
    sensitivity score for the parameter around the different parameter sets tested
"""
import warnings
from functools import partial
from typing import List

import numpy as np
import pandas as pd

from ...misc import blab, par_eval
from ..IO import (
    DigesterFeed,
    DigesterInformation,
    DigesterParameter,
    DigesterState,
    DigesterStates,
    FreeDigesterParameter,
    param_range,
)
from ..prediction_error import adm1_err, score_param
from ..run_adm1 import run_adm1
from ._helper_sensitivity import generate_morris_lines, param_list

n_param = len(param_list)

# Local sensitivity functions
def _eval_param(
    param_dim: int,
    param: DigesterParameter,
    ref_y: DigesterStates,
    influent_state: DigesterFeed,
    initial_state: DigesterState,
    digester_info: DigesterInformation,
    increment: float = 0.01,
    silent: bool = False,
    **kwargs,
) -> float:
    """
    Evaluate the impact of a parameter dimension param_dim by computing the impact of a
        perturbation of this parameter dimenision on predictions.

    Perturbation is performed using a multiplicative factor of exp(increment). Both left
    and right perturbation are evaluated

    Args:
        param_dim: int indicating the dimension to evaluate
        param: the parameter at which to perform the local SA
        ref_y: the reference output (i.e. the output of param)
        influent_state, initial_state, digester_info: see run_adm1 documentation
        increment: the relative increment used. Default is 0.01 (about 1% change)
        silent: should there be any prints?

    This evaluation is parallelized in local_sensitivity to perform a local SA.
    """
    param_mod = param.copy()
    param_mod[param_dim] = param_mod[param_dim] * np.exp(increment)

    impact_dim = score_param(
        param=param_mod,
        obs=ref_y,
        influent_state=influent_state,
        initial_state=initial_state,
        digester_info=digester_info,
        **kwargs,
    )

    param_mod[param_dim] = param_mod[param_dim] * np.exp(-2 * increment)

    impact_dim += score_param(
        param=param_mod,
        obs=ref_y,
        influent_state=influent_state,
        initial_state=initial_state,
        digester_info=digester_info,
        **kwargs,
    )

    impact_dim = impact_dim / (2 * increment)
    blab(silent, f"{param_list[param_dim]}: {impact_dim}")
    return impact_dim


def local_sensitivity(
    param: DigesterParameter,
    influent_state: DigesterFeed,
    initial_state: DigesterState,
    digester_info: DigesterInformation,
    increment: float = 0.01,
    max_step: float = 15 / (24 * 60),
    parallel: bool = True,
    silent: bool = False,
    **kwargs,
) -> pd.Series:
    r"""Local sensitivity estimation

     Each parameter is scored through
     Score_i = adm1_err(ADM1(theta_i * np.exp(increment)), ADM1(theta))/(increment)
     (This should have a limit when increment goes to 0)

     Args:
         param, parameter around which to compute sensitivity
         influent_state, intial_state, digester_info -> see run_adm1 doc
         increment, controls the modification applied to each parameter.
             If increment is small, it amounts to a relative change of increment.
             Default is 0.01 (about 1% change).
         parallel: bool, should evaluations of parameter impact be parallelized (Default is True)
         silent: bool, should there be no prints (default is False)
     kwargs:
         passed to run_adm1

     Returns:
         A pandas.Series giving a score to each parameter. The higher the score,
         the higher the impact this parameter has on the output.

     Score interpretation: a score of S_i for parameter i implies that modifying the values of
     parameter i by $\alpha$ percent will have an average relative change on the output values of
    $\alpha * S_i$ when $\alpha * S_i \ll 1$.
    """
    blab(silent, "\n============================\n", "Sensitivity Analysis - Local\n")

    # Reference value
    ref_y = run_adm1(
        param,
        influent_state=influent_state,
        initial_state=initial_state,
        digester_info=digester_info,
        **kwargs,
    )

    loc_evaluate_param = partial(
        _eval_param,
        param=param,
        increment=increment,
        ref_y=ref_y,
        influent_state=influent_state,
        initial_state=initial_state,
        digester_info=digester_info,
        silent=silent,
        max_step=max_step,
    )

    par_sensit = pd.Series(
        par_eval(loc_evaluate_param, range(n_param), parallel=parallel),
        index=param_list,
    )
    return par_sensit


# Global sensitivity functions
def evaluate_morris_line(
    line_params: List[FreeDigesterParameter],
    influent_state: DigesterFeed,
    initial_state: DigesterState,
    digester_info: DigesterInformation,
    silent: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Evaluate morris line by computing error between two successive parameters.

    Args:
        line_param, a list of FreeDigesterParameter (In fact, any iterable of ArrayLike convertible to DigesterParameter
            should work)
    Find a rule to deal with numerical errors (nans, inf.)
    """
    accu = np.zeros(n_param)
    param_ini = FreeDigesterParameter.to_dig_param(line_params[0])  # More robust
    pred_ini = run_adm1(
        param_ini, influent_state, initial_state, digester_info, **kwargs
    )

    for i in range(n_param):
        param_new = FreeDigesterParameter.to_dig_param(line_params[i + 1])
        try:
            pred_new = run_adm1(param_new, influent_state, initial_state, digester_info)
        except Exception as exc:  # pylint: disable=W0703
            warnings.warn(f"run_adm1 failed for parameter {param_new}: {exc}")
            pred_new = pred_ini
            err = 2
        else:
            err = adm1_err(pred_new, pred_ini)
        accu[i] = err
        blab(silent, f"{param_list[i]}: {err}")
        pred_ini = pred_new
    return accu


def global_sensitivity(
    influent_state: DigesterFeed,
    initial_state: DigesterState,
    digester_info: DigesterInformation,
    r: int = 10,
    n_lev: int = 10,
    param_range: pd.DataFrame = param_range,  # pylint: disable=W0621
    parallel: bool = True,
    silent: bool = False,
) -> pd.DataFrame:
    r"""
    Morris-like global, parameter wise sensitivity analysis.

    Args:
        influent_state, initial_state, digester_info -> see run_adm1 doc
        r, number of lines built.
        n_lev, number of levels considered for each type of parameters. A regular grid in log space is used.
        param_range, range for each parameter. Should be formatted as a pd.DataFrame with columns "min" and "max"
            and rows indexed with parameter names. param_range is coded for FreeDigesterParameter rather than DigesterParameter,
            i.e., the min describes the minimum value that the shifted log of the parameter value can attain
            (and typically is negative since 0 is mapped to the default parameter value).

    The default param_range is obtained using the default values from Rosen & Jeppson and the initial report for ADM1 for the
    default variability on each parameter. Levels of uncertainty 1, 2 and 3 were respectively transformed into standard deviations for
    the log parameters of .12, .4 and 1.2, and the range for parameter i is defined as
            $[default_i * \exp(-2 * \sigma_i), default_i * \exp(2 * \sigma_i)]$

    Returns:
        A pandas.DataFrame with columns the parameters and index "mean" and "sd", giving respectively the mean and standard deviation of the
        sensitivity scores for each type of parameter.

    """
    blab(silent, "\n============================\n", "Sensitivity Analysis - Global\n")

    ## Prepare lines
    blab(silent, "Prepare morris lines")
    morris_lines = generate_morris_lines(r=r, n_lev=n_lev, param_range=param_range)
    blab(silent, "Evaluation of morris lines")
    loc_evaluate_morris = partial(
        evaluate_morris_line,
        influent_state=influent_state,
        initial_state=initial_state,
        digester_info=digester_info,
        silent=silent,
    )
    # Parallelize that
    evals = np.array(par_eval(loc_evaluate_morris, morris_lines, parallel=parallel))

    # evals = [loc_evaluate_morris(morris_line) for morris_line in morris_lines]
    ## evals is a 2D array. column k is the impact of parameter k
    means = [n_lev * np.nanmean(np.abs(impact)) for impact in evals.T]

    devs = [n_lev * np.nanstd(np.abs(impact)) for impact in evals.T]
    blab(silent, "\nSensitivity Analysis completed\n", "==============\n")
    return pd.DataFrame([means, devs], columns=param_list, index=["mean", "sd"])
