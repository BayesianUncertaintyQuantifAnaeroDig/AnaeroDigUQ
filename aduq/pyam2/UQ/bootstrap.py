from typing import List, Optional, Type

import numpy as np

from ...misc import blab, par_eval
from ...uncertainty import lin_bootstrap
from .._typing import DigesterFeed, DigesterParameter, DigesterState, DigesterStates
from ..der_am2 import am2_derivative
from ..IO import param_names_to_index, parameter_dict
from ..optim import optim_am2
from ..optim._helper import default_cov
from ..run_am2 import run_am2


def am2_lin_bootstrap(
    n_boot: int,
    opti_param: DigesterParameter,
    obs: DigesterStates,
    params_eval: Optional[list[str]] = None,
    pred_gradient: Optional[np.ndarray] = None,
    influent_state: Optional[DigesterFeed] = None,
    initial_state: Optional[DigesterState] = None,
    param_output: Optional[DigesterStates] = None,
    in_log: Optional[bool] = True,
    silent: bool = False,
    **kwargs,
) -> dict:
    """Linear bootstrap procedure in AM2 context

    Args:
        n_boot: number of bootstrapped parameters to generate
        obs: observations
        opti_param: calibrated parameter on original data
        params_eval: names of parameters calibrated
        pred_gradient: Optional gradient of the prediction (either in log or std depending on
            in_log). If None, computed from remaining parameters
        influent_state, initial_state: Optional, see run_am2 documentation. If any
            of those is None, then both pred_gradient and param_output must be specified
        param_output: predictions obtained using the calibrated parameters.
        in_log: Should the procedure use std or log residuals?
        silent: Should there be no print?

    Output:
        a dictionnary with keys 'sample', 'pred_gradient' and 'param_output'.
        sample: np.ndarray containing the bootstrapped samples (shape (n_boot, len(opti_param)))
        pred_gradient: gradient of run_am2 (or log(run_am2) if in_log)
        param_output: output of run_am2

    Future:
        Consider create a class to be able to cache param_output and pred_gradient more efficiently
    """
    blab(silent, "Starting linear bootstrap procedure")

    if params_eval is None:
        params_eval = list(parameter_dict)

    # Assess missing input
    can_simul = not ((influent_state is None) or (initial_state is None))

    # Check that either param_output is given or can be computed
    if param_output is None:
        if not can_simul:
            raise ValueError("Predictions can not be computed")
        blab(silent, "Computing output of AM2 model")
        param_output = run_am2(opti_param, influent_state, initial_state, **kwargs)

    # Check that either pred_gradient is given or can be computed
    if pred_gradient is None:
        if not can_simul:
            raise ValueError("Gradient can not be computed")
        blab(silent, "Computing gradient of AM2 model")
        pred_gradient = am2_derivative(
            param=opti_param,
            influent_state=influent_state,
            initial_state=initial_state,
            params_to_der=params_eval,
            log_am2=in_log,
            am2_out=param_output,
            **kwargs,
        )

    if in_log:
        res = np.log(obs[:, 1:]) - np.log(param_output[:, 1:])
    else:
        res = obs[:, 1:] - param_output[:, 1:]

    blab(silent, "Generating the bootstrapped samples")
    red_boot_samples = lin_bootstrap(
        res=res,
        grad=pred_gradient,
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


def am2_bootstrap(
    n_boot: int,
    opti_param: DigesterParameter,
    obs: DigesterStates,
    influent_state: DigesterFeed,
    initial_state: DigesterState,
    param_output: Optional[DigesterParameter] = None,
    # Optimisation method
    optim_method: str = "CMA",
    # Termination criteria
    chain_length: int = 25,
    xtol: float = 10 ** (-4),
    ftol: float = 10 ** (-4),
    # Main arguments for CMA-ES
    cov_ini: Type[np.ndarray] = default_cov,
    radius_ini: float = 0.01,
    # Main arguments for MH
    # Further secondary arguments
    per_step: Optional[int] = None,
    no_change_max: int = 10,
    radius_factor: float = 0.8,
    cov_updt_speed: float = 0.95,
    keep_frac: float = 0.25,
    # Parallel/print
    parallel: bool = True,
    parallel_in_optim: bool = False,
    print_rec: int = 5,
    silent: bool = False,
    **kwargs,
) -> List[DigesterParameter]:
    """
    Bootstrap method for uncertainty quantification
    After optimisation, residuals are bootstrapped and generate new train data
    Different optimisatio procedures n are then run. This defines a sample of parameter,
    which is then used for parameter uncertainty quantification.

    Parallelisation resolution:
        parallel states whether parallel computing should be used at all
        parallel_in_optim state whether each optimisation procedure should use
            parallel computing (i.e. optimisations are done sequentially, each using all cores)
        If parallel_in_optim is False (defaults), parallelisation is done (if at all) above the
            optimisations (i.e. optimisations are done in parallel, each using one core).

        Note that if parallel is False, no parallelisation is done, regardless of parallel_in_optim

        The default should be more efficient.
    """
    if param_output is None:
        param_output = run_am2(
            param=opti_param,
            influent_state=influent_state,
            initial_state=initial_state,
            **kwargs,
        )

    if not parallel:
        parallel_in_optim = False
        par_across = False
    else:
        par_across = not parallel_in_optim

    def calib_am2(n: int):
        blab(silent, f"Starting bootstrapped calibration {n+1}")
        np.random.seed(n)  # Avoid issue with parallel computing!
        obs_boot = bootstrap_res(obs, param_output)
        param_fitted = optim_am2(
            init_param=opti_param,
            obs=obs_boot,
            influent_state=influent_state,
            initial_state=initial_state,
            chain_length=chain_length,
            xtol=xtol,
            ftol=ftol,
            optim_method=optim_method,
            cov_ini=cov_ini,
            radius_ini=radius_ini,
            per_step=per_step,
            no_change_max=no_change_max,
            radius_factor=radius_factor,
            cov_updt_speed=cov_updt_speed,
            keep_frac=keep_frac,
            parallel=parallel_in_optim,
            print_rec=print_rec,
            silent=silent,
            **kwargs,
        ).opti_param
        return param_fitted

    return par_eval(calib_am2, np.arange(n_boot), par_across)
