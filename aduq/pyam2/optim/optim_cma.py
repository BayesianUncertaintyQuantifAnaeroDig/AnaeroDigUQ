from functools import partial
from typing import Optional, Type

import numpy as np

from ...optim import OptimResultCMA, optim_CMA
from .._typing import DigesterFeed, DigesterParameter, DigesterState, DigesterStates
from ..interp_param import interp_param, inv_par_map, par_map
from ..prediction_error import score_param
from ._helper import default_cov


def optim_cma_am2(
    init_param: DigesterParameter,
    # Computation of score
    obs: DigesterStates,
    influent_state: DigesterFeed,
    initial_state: DigesterState,
    # Termination arguments
    chain_length: int = 10,
    xtol: float = 10 ** (-6),
    ftol: float = 10 ** (-6),
    # Main arguments
    cov_ini: Type[np.ndarray] = default_cov,
    radius_ini: float = 1.0,
    # Further secondary arguments
    per_step: Optional[int] = None,
    no_change_max: int = 10,
    radius_factor: float = 0.9,
    cov_updt_speed: float = 0.95,
    keep_frac: float = 0.25,
    parallel: bool = True,
    print_rec: int = 5,
    silent: bool = False,
    **kwargs,
) -> OptimResultCMA:
    """
    CMA-ES optimization routine for AM2

    Args:
        init_param: initial parameter (as a DigesterParameter)
        obs: observations
        influent_state, initial_state: passed to run_am2
        chain_length: maximum number of optimisation iteration
        xtol, ftol: termination criteria for the parameter and score respectively
        cov_ini: Covariance in FREE DIGESTER PARAMETER space. Must have covariance information for all parameters.
        radius_ini: initial radius adjustement on covariance used to draw proposal parameters (effective covariance is radius_ini**2 * cov_ini)
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
        further arguments passed to run_am2 and am2_err

    Outputs:
    Outputs:
        OptimResultCMA object with keys opti_param, opti_score, converged, hist_param, hist_score, hist_cov,
        Parameters are in DigesterParameter space.

    - opti_param is a DigesterParameter, accu_param contains DigesterParameter
    - accu_cov contains covariance in log space (i.e. needs to use par_map to transform
        uncertainty to parameter)

    """
    # DigesterParameter -> Free
    start_param = inv_par_map(init_param)
    score = interp_param(
        partial(
            score_param,
            obs=obs,
            influent_state=influent_state,
            initial_state=initial_state,
            **kwargs,
        )
    )

    # Main call to CMA_optim routine
    opt_res = optim_CMA(
        param_ini=start_param,
        score=score,
        xtol=xtol,
        ftol=ftol,
        chain_length=chain_length,
        cov_ini=cov_ini,
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

    # Convert back to DigesterParameter
    opt_res.convert(par_map)

    return opt_res
