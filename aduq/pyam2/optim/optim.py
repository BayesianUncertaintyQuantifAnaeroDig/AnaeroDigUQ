""" Wrap up optimizer function.

--- Maintenance rules ---
Argument order:
- Start parameter first in DigesterParameter space
- Score computation related arguments obs + AM2 main arguments (secondary arguments can be passed
    as **kwargs)
- Space on which to optimize (params_eval)
- Termination criteria
- Optimisation routine id
- CMA-ES arguments
- MH arguments
- Future optimizer arguments

As far as possible, try to share arguments names between optimisation methods when similar

Optimizer outputs:
    A OptimResult object with keys opti_param, opti_score, converged, hist_param, hist_score (+ optim specific keys)
 """

from typing import Optional, Type

import numpy as np

from ...optim import OptimResult
from .._typing import DigesterFeed, DigesterParameter, DigesterState, DigesterStates
from ._helper import default_cov
from .optim_cma import optim_cma_am2
from .optim_mh import optim_mh_am2


def optim_am2(
    init_param: DigesterParameter,
    # Computation of score
    obs: DigesterStates,
    influent_state: DigesterFeed,
    initial_state: DigesterState,
    # Termination criteria
    chain_length: int = 100,
    xtol: float = 10 ** (-6),
    ftol: float = 10 ** (-6),
    # Optimisation method
    optim_method: str = "CMA",
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
    parallel: bool = True,
    print_rec: int = 5,
    silent: bool = False,
    **kwargs,
) -> Type[OptimResult]:
    """
    Calibration of AM2

    Search for the parameter best describing the digester dynamics.
    The error is computed as, mostly,
    RMSE(log(AM2(parameter)/ obs)).
    For full details about the error implementation, see adm1_err documentation

    Optimisation routines proposed are: {"CMA", "MH"}. "CMA" uses CMA-ES algorithm,
    while "MH" uses a Metropolis-Hastings inspired optimisation algorithm.
    Both techniques draw parameter iteratively around the parameter achieveing the best score so far.
    For CMA-ES, the covariance used to draw parameters is modified,
    while for MH, the covariance structure is fixed and only its radius is modified (cov_t = r_t **2 * cov_ini).
    Both are semi-local optimisation routines. For global optimisation, multiple start procedure are advised.

    Optimisation is done using FreeDigesterParameter under the hood.
    Inputs and outputs should be DigesterParameter, EXCEPT cov_ini and radius_ini which are used to define
    the proposal covariance in the FreeDigesterParameter space.


    Args:
        init_param: initial DigesterParameter
        obs: DigesterStates, the observed data used as calibration target
        influent_state, initial_state: arguments for run_am2 routine, see run_am2 doc
        params_eval: list of parameters names which should be optimized. Default is None, amounting to all parameters
        chain_length: maximum number of optimization steps. Default is 100.
        xtol: termination criteria on parameter (approx. uncertainty on parameter)
        ftol: termination criteria on error (approx. uncertainty on minimum score)
        optim_method: optimisation method used (either "CMA" or "MH")
        cov_ini: initial covariance used to draw proposal parameters. The covariance is defined for FreeDigesterParameter
        radius_ini: initial radius adjustement on covariance used to draw proposal parameters (effective covariance is radius_ini**2 * cov_ini)
        per_step! number of proposal parameters drawn at each optimiation step
        no_change_max: number of optimisation step without finding a better parameter before the covariance radius is contracted
        radius_factor: factor by which the covariance radius is contracted (cov_updt = radius_factor **2 * cov_prev)
        cov_updt_speed: controls the speed at which the covariance is updated (for CMA-ES algorithm)
        keep_frac: fraction of proposal parameter kept to modify covariance (for CMA-ES algorithm)
        parallel: should the error evaluations be parallelized?
        print_rec: prints optimisation information every print_rec optimisation steps
        silent: if silent, does not print anything (except warnings/errors)

    kwargs are passed to score_param and include both further arguments for score_param and run_am2.

    Outputs an OptimResult object, with attributes including opti_param, converged, opti_score, hist_param, hist_score
    """
    if optim_method == "CMA":
        return optim_cma_am2(
            init_param=init_param,
            obs=obs,
            influent_state=influent_state,
            initial_state=initial_state,
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
            print_rec=print_rec,
            silent=silent,
            **kwargs,
        )

    if optim_method == "MH":
        return optim_mh_am2(
            init_param=init_param,
            obs=obs,
            influent_state=influent_state,
            initial_state=initial_state,
            chain_length=chain_length,
            xtol=xtol,
            ftol=ftol,
            cov_ini=cov_ini,
            radius_ini=radius_ini,
            # Further secondary arguments
            per_step=per_step,
            no_change_max=no_change_max,
            radius_factor=radius_factor,
            parallel=parallel,
            print_rec=print_rec,
            silent=silent,
            **kwargs,
        )

    raise Exception(f"{optim_method} is not implemented (choose between 'CMA' and 'MH)")
