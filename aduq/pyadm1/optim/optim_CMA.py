from functools import partial
from typing import Optional, Type

import numpy as np

from ...misc import interpretation, post_modif
from ...optim import OptimResultCMA, optim_CMA
from ..IO import (
    DigesterFeed,
    DigesterInformation,
    DigesterParameter,
    DigesterState,
    DigesterStates,
    free_param,
    free_to_param,
)
from ..IO._helper_pd_np import param_names_to_index
from ..IO.dig_param import bound_param
from ..prediction_error import score_param
from ..proba import default_proposal_cov


def optim_cma_adm1(
    init_param: DigesterParameter,
    # Computation of score
    obs: DigesterStates,
    influent_state: DigesterFeed,
    initial_state: DigesterState,
    digester_info: DigesterInformation,
    # Space on which to compute score
    params_eval: Optional[list[str]] = None,
    force_bound: bool = True,
    # Termination parameters
    chain_length: int = 1,
    xtol: float = 10 ** (-8),
    ftol: float = 10 ** (-8),
    # Main arguments
    cov_ini: Type[np.ndarray] = default_proposal_cov,
    radius_ini: float = 0.01,
    # Further secondary arguments
    per_step: Optional[int] = None,
    no_change_max: int = 10,
    radius_factor: float = 0.8,
    cov_updt_speed: float = 0.95,
    keep_frac: float = 0.25,
    # Global arguments
    parallel: bool = True,
    print_rec: int = 1,
    silent: bool = False,
    **kwargs,
) -> OptimResultCMA:
    """
    CMA-ES optimization routine for ADM1. Optimisation is done in FreeDigesterParameter space (which is unconstrained).
    Conversion is done inside the function.

    Args:
        init_param: initial parameter (as a DigesterParameter)
        obs: observations
        influent_state, initial_state, digester_info: passed to run_adm1,
        params_eval: list of parameter names, on which optimisation is to occur (remaining
            parameters are fixed)
        force_bound: should the optimisation task search for parameters with upper bounds? Default is True

        chain_length: maximum number of optimisation iteration
        xtol, ftol: termination criteria for the parameter and score respectively

        cov_ini: Covariance in FREE DIGESTER PARAMETER space. Must have covariance information for
            all parameters.
        radius_ini: initial radius adjustement on covariance used to draw proposal parameters
            (effective covariance is radius_ini**2 * cov_ini)
        per_step! number of proposal parameters drawn at each optimiation step
        no_change_max: number of optimisation step without finding a better parameter before the
            covariance radius is contracted
        radius_factor: factor by which the covariance radius is contracted
            (cov_updt = radius_factor **2 * cov_prev)
        cov_updt_speed: controls the speed at which the covariance is updated
        keep_frac: fraction of proposal parameter kept to modify covariance
        parallel: should the error evaluations be parallelized?
        print_rec: prints optimisation information every print_rec optimisation steps
        silent: if silent, does not print anything (except warnings/errors)
    kwargs:
        further arguments passed to run_adm1 and adm1_err

    Outputs:
        OptimResultCMA object with keys opti_param, opti_score, converged, hist_param, hist_score, hist_cov,
        Parameters are in DigesterParameter space.

    """
    # DigesterParameter -> Free
    start_param = free_param(init_param)

    # Set up interpreter of reduced free parameter
    if params_eval is not None:

        params_index = param_names_to_index(params_eval)
        ref = start_param.copy()

        def pre_set_up_param(x: np.ndarray) -> DigesterParameter:
            """Transform a small parameter into a full parameter"""
            par = ref.copy()
            par[params_index] = x
            return free_to_param(par)

        small_cov = cov_ini[params_index][:, params_index]
        start_param = np.array(start_param[params_index])

    else:
        pre_set_up_param = free_to_param
        small_cov = cov_ini

    # Force bound the parameters
    if force_bound:
        set_up_param = post_modif(bound_param)(pre_set_up_param)
    else:
        set_up_param = pre_set_up_param

    loc_interp = interpretation(set_up_param)

    score = loc_interp(
        partial(
            score_param,
            obs=obs,
            influent_state=influent_state,
            initial_state=initial_state,
            digester_info=digester_info,
            **kwargs,
        )
    )

    opt_res = optim_CMA(
        param_ini=start_param,
        score=score,
        xtol=xtol,
        ftol=ftol,
        chain_length=chain_length,
        cov_ini=small_cov,
        radius_ini=radius_ini,
        per_step=per_step,
        no_change_max=no_change_max,
        radius_factor=radius_factor,
        cov_updt_speed=cov_updt_speed,
        keep_frac=keep_frac,
        parallel=parallel,
        vectorized=False,
        print_rec=print_rec,
        silent=silent,
    )

    opt_res.convert(set_up_param)

    return opt_res
