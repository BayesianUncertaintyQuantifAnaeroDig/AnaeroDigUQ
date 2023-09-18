"""
beale's UQ routine for ADM1.

Relies on main implementation of beale's UQ method in 'uncertainty' module.
"""
import warnings
from functools import partial
from typing import List

import numpy as np
import pandas as pd

from ...misc import blab, par_eval
from ...uncertainty import beale_boundary, beale_pval
from ..IO import (
    DigesterFeed,
    DigesterInformation,
    DigesterParameter,
    DigesterState,
    DigesterStates,
    parameter_dict,
    small_predictions_numb,
)
from ..prediction_error import ADM1Failure, score_param
from ..run_adm1 import run_adm1
from .pred_uncertainty import prediction_uncertainty_from_output


class RemovedAll(Exception):
    """Custom class for cases where pre-processing results in an empty problem"""


def _score(
    param: np.ndarray,
    param_ref: DigesterParameter,
    params_eval_index: List[int],
    obs: DigesterStates,
    influent_state: DigesterFeed,
    initial_state: DigesterState,
    digester_info: DigesterInformation,
    max_score: float = 20.0,
    **kwargs,
) -> dict:
    """
    Function used to compute score of small parameter given in log space.
    True parameter is prepared, then used.
    """
    # prepare parameter
    param_loc = param_ref.copy()
    param_loc[params_eval_index] = np.exp(param)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=ADM1Failure)
        return (
            score_param(
                param_loc,
                obs=obs,
                influent_state=influent_state,
                initial_state=initial_state,
                digester_info=digester_info,
                max_score=max_score,
                **kwargs,
            )
            ** 2
        )


def adm1_beale(
    n_boundary: int,
    conf_lev: float,
    cov: pd.DataFrame,
    param: DigesterParameter,
    params_eval: list,
    obs: DigesterStates,
    influent_state: DigesterFeed,
    initial_state: DigesterState,
    digester_info: DigesterInformation,
    max_iter: int = 20,
    max_time: int = 600,
    relat_crit: float = 0.001,
    max_log_dev: float = 6.0,
    parallel: bool = True,
    silent: bool = False,
    **kwargs,
) -> dict:  # Could be changed to list of DigesterParameter
    """
    Wrap up of beale_boundary used to do conversions + prepare ADM1 function.

    ADM1 modification:
        Date is removed from ADM1 output
        Only parameters specified in params_eval are modified
        Input is in log space

    args:
        n_boundary: the number of points to compute in the boundary of the uncertainty region
        conf_lev: the confidence level of the uncertainty region to compute,
        cov: approximative covariance matrix for the parameters
        param: the optimal parameter found
        params_eval: the parameters on which the uncertainty quantification is conducted
        influent_state, initial_state, digester_info, solver_method -> arguments passed to run_adm1
        max_iter: Maximum number of iterations in line search. Default is 20
        max_time: Maximum time (in seconds) for evaluation of ADM1 model. Default is 600 s.
        relat_crit: Relative precision of the line search before convergence. Default is 0.001.
            This can be overridden if too large:
            should be smaller than 0.05 *  ((score_threshold/ score_min) - 1).
        max_log_dev: Filter on parameters with deviations in log space larger than max_log_dev.
            Default is 6.0.
        parallel: Should line searches be parallelized? Default is True.
        silent: Should the function not print intermediary information? Default is False.
    """
    # Starting print statement
    print(
        "\n====================================\n",
        "Uncertainty quantification through error bound\n",
        sep="\n",
    )

    # Interpret defaults
    blab(silent, f"Optimal parameter:\n{param.to_pandas()[params_eval]}")

    # Define covariance matrix on parameters to evaluate
    cov.index = cov.columns
    cov = cov.loc[params_eval, params_eval]

    blab(silent, f"Parameter covariance:\n{cov}")
    cov = cov.to_numpy()

    ## Translate covariance matrix to log space
    param_nump = param.to_pandas()[params_eval].to_numpy()
    cov_log = (cov * (param_nump ** (-1))).T * (param_nump ** (-1))

    cov_log = pd.DataFrame(cov_log, columns=params_eval, index=params_eval)
    blab(silent, f"Parameter covariance in log space:\n{cov_log}")

    # Necessary to filter on parameters which have too large uncertainty
    # else numerical errors happen too frequently
    std_log_param = np.sqrt(np.diag(cov_log))
    blab(silent, f"Standard deviation in log space: {std_log_param}")
    ok_param = std_log_param < max_log_dev

    # Inform if any parameter is removed
    if np.sum(~ok_param) > 0:

        # Check that not all parameters are removed
        # Not that if only a single parameter remains, boundary could be 2 points
        if np.all(~ok_param):
            raise RemovedAll("All parameters are deemed to have excessive uncertainty")

        warnings.warn(
            "\n".join(
                [
                    "Uncertainty quantification is impossible for the following parameters:",
                    f"{[par for par, ok in zip(params_eval, ok_param) if not ok]}",
                ]
            )
        )
        # Remove parameters with too large deviations
        params_eval = [par for par, ok in zip(params_eval, ok_param) if ok]
        cov_log = cov_log[ok_param][:, ok_param]

    # Prepare reference parameter
    param_full = param.copy()
    params_eval_index = [
        parameter_dict[param_name] for param_name in params_eval
    ]  # pd.Series -> list of int

    # Prepare score function
    score = partial(
        _score,
        param_ref=param_full,
        params_eval_index=params_eval_index,
        obs=obs,
        influent_state=influent_state,
        initial_state=initial_state,
        digester_info=digester_info,
        **kwargs,
    )

    # Compute reduced optimal parameter
    opti_param = np.array(np.log(param_full[params_eval_index]))

    # Compute number of observations
    n_obs = int(np.sum(~np.isnan(obs[:, small_predictions_numb[1:]])))

    # Optional print before passing to main function (help for debugging)
    blab(
        silent,
        "Calling beale_boundary function.",
        "Confidence region boundary is computed in log-parameters space.",
        "Optimal parameter in this space is:",
        f"{np.asarray(np.log(param_full[params_eval_index]))}",
        "Covariance is:",
        f"{cov_log}\n",
        f"Confidence level: {conf_lev}",
        f"Points on the boundary: {n_boundary}",
        "------------------",
        sep="\n",
    )

    # Main function call
    to_return = beale_boundary(
        opti_param=opti_param,
        score_func=score,
        cov=cov_log,
        n_obs=n_obs,
        conf_lev=conf_lev,
        n_boundary=n_boundary,
        max_iter=max_iter,
        max_time=max_time,
        relat_crit=relat_crit,
        parallel=parallel,
        silent=silent,
    )

    # Converting the boundary to standard place + translating as pd.DataFrame
    to_return["boundary"] = pd.DataFrame(
        np.exp(to_return["boundary"]), columns=params_eval
    )

    # Add information on which parameters were evaluated
    to_return["params_eval"] = params_eval

    # Additional informations
    to_return["score"] = score
    to_return["conf_lev"] = conf_lev
    to_return["n_obs"] = n_obs

    return to_return


def adm1_beale_pval(
    param: DigesterParameter,
    opti_param: DigesterParameter,
    n_params: int,
    obs: DigesterStates,
    influent_state: DigesterFeed,
    initial_state: DigesterState,
    digester_info: DigesterInformation,
    silent: bool = True,
    **kwargs,
) -> float:
    """Compute minimum confidence level necessary to cover a parameter param using Beale
    uncertainty technique

    Args:
        - param, a digester parameter for which we want to compute required coverage
        - opti_param, the parameter obtained through error minimisation
        - obs, the observation used for calibration
        - n_params, the number of parameters fitted

    Output:
        a float giving the p-value of the hypothesis that the data was generated by param
        computed through an F statistic (see uncertainty.beale documentation).
    """
    blab(
        silent,
        """
    Scoring beale uncertainty
    -------------------------
    """,
    )
    blab(
        silent,
        f"Parameter to evaluate:\n{param}\n",
        f"Optimal parameter:\n{opti_param}\n",
        sep="\n",
    )

    n_obs = int(np.sum(~np.isnan(obs[:, small_predictions_numb[1:]])))

    score = partial(
        score_param,
        obs=obs,
        influent_state=influent_state,
        initial_state=initial_state,
        digester_info=digester_info,
        **kwargs,
    )

    score_ev, score_opti = score(param) ** 2, score(opti_param) ** 2
    blab(
        silent,
        f"Score of parameter:{score_ev}\nScore of optimal parameter:{score_opti}",
    )
    return beale_pval(
        n_param=n_params, n_obs=n_obs, score_opt=score_opti, score_param=score_ev
    )


def beale_prediction_uncertainty(
    boundary: List[DigesterParameter],
    opti_param: DigesterParameter,
    influent_state,
    initial_state,
    digester_info,
) -> List[DigesterStates]:
    """
    Evaluate prediction uncertainty from a log of predictions using beale's method.

    Add elements inside the boundary
    """
    loc_adm1 = partial(
        run_adm1,
        influent_state=influent_state,
        initial_state=initial_state,
        digester_info=digester_info,
    )

    # Add points inside credibility region
    inside_points = [(param + opti_param) / 2 for param in boundary]
    boundary.extend(inside_points)
    boundary.append(opti_param)

    predictions = par_eval(loc_adm1, boundary, parallel=True)

    param_UQ_transfer = prediction_uncertainty_from_output(predictions, min_max=True)
    return param_UQ_transfer
