from typing import List, Optional

import numpy as np

from ...misc import blab, par_eval
from ...uncertainty import lin_bootstrap
from ..der_adm1 import adm1_derivative
from ..IO import (
    DigesterFeed,
    DigesterInformation,
    DigesterParameter,
    DigesterState,
    DigesterStates,
    param_names_to_index,
    small_predictions_numb,
)
from ..optim import optim_adm1
from ..run_adm1 import run_adm1


def adm1_lin_bootstrap(
    n_boot: int,
    opti_param: DigesterParameter,
    obs: DigesterStates,
    params_eval: list[str],
    pred_gradient: Optional[np.ndarray] = None,
    influent_state: Optional[DigesterFeed] = None,
    initial_state: Optional[DigesterState] = None,
    digester_info: Optional[DigesterInformation] = None,
    param_output: Optional[DigesterStates] = None,
    in_log: Optional[bool] = True,
    silent: bool = False,
    **kwargs,
) -> dict:
    """Linear bootstrap procedure in ADM1 context

    Args:
        n_boot: number of bootstrapped parameters to generate
        obs: observations
        opti_param: calibrated parameter on original data
        params_eval: names of parameters calibrated
        pred_gradient: Optional gradient of the prediction (either in log or std depending on
            in_log). If None, computed from remaining parameters
        influent_state, initial_state, digester_info: Optional, see run_adm1 documentation. If any
            of those is None, then both pred_gradient and param_output must be specified
        param_output: predictions obtained using the calibrated parameters.
        in_log: Should the procedure use std or log residuals?
        silent: Should there be no print?

    Output:
        a dictionnary with keys 'sample', 'pred_gradient' and 'param_output'.
        sample: np.ndarray containing the bootstrapped samples (shape (n_boot, len(opti_param)))
        pred_gradient: gradient of run_adm1 (or log(run_adm1) if in_log)
        param_output: output of run_adm1

    Future:
        Consider create a class to be able to cache param_output and pred_gradient more efficiently
    """
    blab(silent, "Starting linear bootstrap procedure")
    # Assess missing input
    can_simul = not (
        (influent_state is None) or (initial_state is None) or (digester_info is None)
    )

    # Check that either param_output is given or can be computed
    if param_output is None:
        if not can_simul:
            raise ValueError("Predictions can not be computed")
        blab(silent, "Computing output of ADM1 model")
        param_output = run_adm1(
            opti_param, influent_state, initial_state, digester_info, **kwargs
        )

    # Check that either pred_gradient is given or can be computed
    if pred_gradient is None:
        if not can_simul:
            raise ValueError("Gradient can not be computed")
        blab(silent, "Computing gradient of ADM1 model")
        pred_gradient = adm1_derivative(
            opti_param,
            influent_state,
            initial_state,
            digester_info,
            params_to_der=params_eval,
            log_adm1=in_log,
            adm1_out=param_output,
            **kwargs,
        )

    err_pred_ind = small_predictions_numb[1:]
    err_pred_correct = [ind - 1 for ind in err_pred_ind]  # Correct for removal of Time

    red_param_output = np.array(param_output[:, 1:])  # Remove time!

    if in_log:
        res = np.log(np.array(obs[:, 1:])) - np.log(red_param_output)
    else:
        res = np.array(obs[:, 1:]) - red_param_output

    blab(silent, "Generating the bootstrapped samples")
    red_boot_samples = lin_bootstrap(
        res[:, err_pred_correct],
        pred_gradient[:, :, err_pred_correct],
        weights=None,
        n_boot=n_boot,
    )

    boot_samples = np.full((n_boot, len(opti_param)), np.array(opti_param))
    ind_params_eval = param_names_to_index(params_eval)
    boot_samples[:, ind_params_eval] = (
        boot_samples[:, ind_params_eval] + red_boot_samples
    )

    return {
        "sample": boot_samples,
        "pred_gradient": pred_gradient,
        "param_output": param_output,
    }


def bootstrap_res(obs: DigesterStates, predictions: DigesterStates) -> DigesterStates:
    """
    From observation and predictions, generate new observations by bootstrapping residuals
    Residuals are bootstrapped in log space

    bootstrap_error should not have any side effect on obs and predictions.

    Arguments:
        obs, a DigesterStates object
        predictions, a DigesterStates object

    Outputs:
        A DigesterStates based on predictions with bootstrapped log-residuals wrt to obs
    """

    # Check time
    if not all(np.isclose(obs[:, 0], predictions[:, 0])):
        raise Exception("Bootstrap failed: Time information does not match.")

    # Define residuals
    residuals = np.log(obs[:, 1:] / predictions[:, 1:])
    boot_res = np.zeros(residuals.shape)

    n = residuals.shape[0]

    # Bootstrap residuals
    for k in range(residuals.shape[1]):
        boot_res[:, k] = np.random.choice(residuals[:, k], n)

    boot_obs = predictions.copy()  # New DigesterStates
    boot_obs[:, 1:] = boot_obs[:, 1:] * np.exp(boot_res)  # Modify the residuals

    return boot_obs


def adm1_bootstrap(
    n_boot: int,
    opti_param: DigesterParameter,
    obs: DigesterStates,
    params_eval: list[str],
    influent_state: DigesterFeed,
    initial_state: DigesterState,
    digester_info: DigesterInformation,
    param_output: Optional[DigesterStates] = None,
    parallel=True,
    silent=False,
    **kwargs,
) -> List[DigesterParameter]:
    """
    Bootstrap method for ADM1 model.

    After optimization, residuals are bootstrapped and generate new train data
    Different optimization are then run. This defines a sample of parameter,
    which is then used for parameter uncertainty quantification.

    Note:
        Using this function is not advised

    Future:
        - Rewrite this using bootstrap function from uncertainty module
        - Deal with **kwargs passed to run_adm1 (extra arguments result in a warning raised by
            scipy caught as an error by run_adm1, need to pass extra arguments to optim_adm1)
    """
    if param_output is None:
        param_output = run_adm1(
            param=opti_param,
            influent_state=influent_state,
            initial_state=initial_state,
            digester_info=digester_info,
        )

    def calib_adm1(n: int):
        blab(silent, f"Starting bootstrapped calibration {n+1}")
        np.random.seed(n)  # Avoid issue with parallel computing!
        obs_boot = bootstrap_res(obs, param_output)
        param_fitted = optim_adm1(
            init_param=opti_param,
            obs=obs_boot,
            params_eval=params_eval,
            influent_state=influent_state,
            initial_state=initial_state,
            digester_info=digester_info,
            **kwargs,
        )
        return param_fitted

    return par_eval(calib_adm1, range(n_boot), parallel=parallel)
