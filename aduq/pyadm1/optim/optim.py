""" Wrap up optimizer function.

--- Maintenance rules ---
Argument order:
- Start parameter first in DigesterParameter space
- Score computation related arguments obs + ADM1 main arguments (secondary arguments can be passed
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
from ..IO import (
    DigesterFeed,
    DigesterInformation,
    DigesterParameter,
    DigesterState,
    DigesterStates,
)
from ..proba import default_proposal_cov
from .optim_CMA import optim_cma_adm1
from .optim_MH import optim_mh_adm1


def optim_adm1(
    init_param: DigesterParameter,
    # Computation of score
    obs: DigesterStates,
    influent_state: DigesterFeed,
    initial_state: DigesterState,
    digester_info: DigesterInformation,
    params_eval: Optional[list] = None,
    force_bound: bool = True,
    # Terminaton criteria
    chain_length: int = 100,
    xtol: float = 10 ** (-8),
    ftol: float = 10 ** (-8),
    # Optimisation method
    optim_method: str = "CMA",
    # Main arguments for CMA
    cov_ini: Type[np.ndarray] = default_proposal_cov,
    radius_ini: float = 0.01,
    # Further secondary arguments
    per_step: Optional[int] = None,
    no_change_max: int = 10,
    radius_factor: float = 0.9,
    cov_updt_speed: float = 0.95,
    keep_frac: float = 0.25,
    # Speed up/print
    parallel: bool = True,
    print_rec: int = 5,
    silent: bool = False,
    # kwargs passed to error computation
    **kwargs,
) -> Type[OptimResult]:
    """
    Calibration of ADM1

    Search for the parameter best describing the digester dynamics.
    The error is computed as, mostly,
    RMSE(log(ADM1(parameter)/ obs)).
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
        influent_state, initial_state, digester_info: arguments for run_adm1 routine, see run_adm1 doc
        params_eval: list of parameters names which should be optimized. Default is None, amounting to all parameters
            force_bound: should the optimisation task search for parameters with upper bounds? Default is True
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

    kwargs are passed to score_param and include both further arguments for score_param and run_adm1.

    Outputs an OptimResult object, with attributes including opti_param, converged, opti_score, hist_param, hist_score
    """

    if optim_method == "CMA":

        out = optim_cma_adm1(
            init_param=init_param,
            obs=obs,
            influent_state=influent_state,
            initial_state=initial_state,
            digester_info=digester_info,
            params_eval=params_eval,
            force_bound=force_bound,
            chain_length=chain_length,
            xtol=xtol,
            ftol=ftol,
            per_step=per_step,
            cov_ini=cov_ini,
            radius_ini=radius_ini,
            radius_factor=radius_factor,
            no_change_max=no_change_max,
            keep_frac=keep_frac,
            cov_updt_speed=cov_updt_speed,
            parallel=parallel,
            print_rec=print_rec,
            silent=silent,
            **kwargs,
        )

        return out

    if optim_method == "MH":

        out = optim_mh_adm1(
            init_param=init_param,
            obs=obs,
            influent_state=influent_state,
            initial_state=initial_state,
            digester_info=digester_info,
            params_eval=params_eval,
            force_bound=force_bound,
            chain_length=chain_length,
            xtol=xtol,
            ftol=ftol,
            per_step=per_step,
            cov_ini=cov_ini,
            prop_radius=radius_ini,
            radius_factor=radius_factor,
            no_change_max=no_change_max,
            parallel=parallel,
            print_rec=print_rec,
            silent=silent,
            **kwargs,
        )

        return out

    raise Exception(
        f"No implementation for optim_method {optim_method}\nSupported methods are 'CMA', 'MH'"
    )
