from functools import partial
from typing import Optional

import numpy as np

from ...misc import interpretation, post_modif
from ...optim import OptimResult, optim_MH
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


def optim_mh_adm1(
    init_param: DigesterParameter,
    obs: DigesterStates,
    influent_state: DigesterFeed,
    initial_state: DigesterState,
    digester_info: DigesterInformation,
    params_eval: Optional[list] = None,
    force_bound: bool = True,
    # Termination arguments
    chain_length: int = 100,
    xtol: float = 10 ** (-8),
    ftol: float = 10 ** (-8),
    # Main arguments
    per_step: Optional[int] = None,
    cov_ini: Optional[np.ndarray] = default_proposal_cov,
    prop_radius: float = 0.01,
    radius_factor: float = 0.9,
    no_change_max: int = 10,
    # Further
    parallel: bool = True,
    print_rec: int = 5,
    silent: bool = False,
    **kwargs,
) -> OptimResult:
    """
    Metropolis Hastings inspired optimization routine for ADM1.
    Optimisation is done in FreeDigesterParameter space (which is unconstrained).
    Conversion is done inside the function.

    Args:
        init_param: initial parameter (as a DigesterParameter)
        obs: observations
        influent_state, initial_state, digester_info: passed to run_adm1,
        params_eval: list of parameter names, on which optimisation is to occur (remaining parameters are fixed)
        force_bound: should the optimisation task search for parameters with upper bounds? Default is True
        chain_length: maximum number of optimisation iteration
        xtol, ftol: termination criteria for the parameter and score respectively

        cov_ini: Covariance for FreeDigesterParameter. Covariance information for all parameters (even those not in params_eval) is necessary.
        radius_ini: initial radius adjustement on covariance used to draw proposal parameters (effective covariance is radius_ini**2 * cov_ini)
        per_step: number of proposal parameters drawn at each optimiation step
        no_change_max: number of optimisation step without finding a better parameter before the covariance radius is contracted
        radius_factor: factor by which the covariance radius is contracted (cov_updt = radius_factor **2 * cov_prev)
        parallel: should the error evaluations be parallelized?
        print_rec: prints optimisation information every print_rec optimisation steps
        silent: if silent, does not print anything (except warnings/errors)
    kwargs:
        further arguments passed to run_adm1 and adm1_err

    Outputs:
        OptimResult object with keys opti_param, opti_score, converged, hist_param, hist_score,
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

    res = optim_MH(
        param_ini=start_param,
        score=score,
        chain_length=chain_length,
        xtol=xtol,
        ftol=ftol,
        per_step=per_step,
        prop_cov=small_cov,
        radius_ini=prop_radius,
        radius_factor=radius_factor,
        no_change_max=no_change_max,
        parallel=parallel,
        vectorized=False,
        silent=silent,
        print_rec=print_rec,
    )

    res.convert(set_up_param)
    return res
