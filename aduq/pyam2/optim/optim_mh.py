from functools import partial
from typing import Optional, Type

import numpy as np

from ...optim import OptimResult, optim_MH
from .._typing import DigesterFeed, DigesterParameter, DigesterState, DigesterStates
from ..interp_param import interp_param, inv_par_map, par_map
from ..prediction_error import score_param
from ._helper import default_cov


def optim_mh_am2(
    init_param: DigesterParameter,
    # Computation of score
    obs: DigesterStates,
    influent_state: DigesterFeed,
    initial_state: DigesterState,
    # Main arguments for CMA-ES
    chain_length: int = 10,
    xtol: float = 10 ** (-6),
    ftol: float = 10 ** (-6),
    cov_ini: Type[np.ndarray] = 0.01 * default_cov,
    radius_ini: float = 1.0,
    # Further secondary arguments
    per_step: Optional[int] = None,
    no_change_max: int = 10,
    radius_factor: float = 0.9,
    parallel: bool = True,
    print_rec: int = 5,
    silent: bool = False,
    **kwargs,
) -> OptimResult:
    """
    CMA-ES optimization routine for AM2

    Args:
        init_param:  initial parameter (as a DigesterParameter)
        obs: observations
        influent_state, initial_state: passed to run_am2
        cov_ini : Covariance used to draw FreeDigesterParameter objects
        radius_in: initial radius adjustement on covariance used to draw proposal parameters (effective covariance is radius_ini**2 * cov_ini)
        per_step: number of proposal parameters drawn at each optimiation step
        no_change_max: number of optimisation step without finding a better parameter before the covariance radius is contracted
        radius_factor: factor by which the covariance radius is contracted (cov_updt = radius_factor **2 * cov_prev)
        parallel: should the error evaluations be parallelized?
        print_rec: prints optimisation information every print_rec optimisation steps
        silent: if silent, does not print anything (except warnings/errors)
    kwargs:
        further arguments passed to run_am2 and am2_err

    Outputs:
        OptimResult object with keys opti_param, opti_score, converged, hist_param, hist_score,
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

    # Main call to optim_MH routine
    opt_res = optim_MH(
        param_ini=start_param,
        score=score,
        chain_length=chain_length,
        xtol=xtol,
        ftol=ftol,
        per_step=per_step,
        prop_cov=cov_ini,
        radius_ini=radius_ini,
        radius_factor=radius_factor,
        no_change_max=no_change_max,
        parallel=parallel,
        vectorized=False,
        silent=silent,
        print_rec=print_rec,
    )

    opt_res.convert(par_map)
    return opt_res
