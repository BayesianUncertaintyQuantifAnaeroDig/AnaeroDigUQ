"""
Optimisation using modified Metropolis Hastings algorithm

Implementation note:
    - par_eval is not used here since opening/closing pools of workers is unnecessary and takes
    time. This is not crucial though if scoring functions are intensive.
"""
from typing import Any, Callable, Optional, Type

import numpy as np
from multiprocess import Pool, cpu_count  # pylint: disable=E0611

from ..misc import blab
from .optim_result import OptimResult


def optim_MH(
    param_ini: Type[np.ndarray],
    score: Callable[[Any], float],
    # Termination criteria
    chain_length: int,
    xtol: float = 10 ** (-8),
    ftol: float = 10 ** (-8),
    # Main arguments
    per_step: Optional[int] = None,
    prop_cov: Optional[Type[np.ndarray]] = None,
    radius_ini: float = 1.0,
    # Secondary arguments
    radius_factor: float = 0.7,
    no_change_max: int = 10,
    parallel: bool = True,
    vectorized: bool = False,
    silent=False,
    print_rec: int = 5,
) -> OptimResult:
    """
    Optimisation method similar to a Metropolis Hastings algorithm, where the proposal is accepted
    iif it lowers the score.

    The proposals are drawned from gaussian distributions. The covariance is contracted along the
    optimisation route to prevent drastic decrease of the probability of drawing better parameters.

    Args:
        param_ini: initial mean parameter
        score: scoring function, to be minimized
        chain_length: maximum length of the chain
        xtol, ftol: criteria for termination (converged when one of the criteria is met)
        per_step: Number of samples generated and evaluated at each step
        prop_cov: Initial covariance structure on parameters
        radius_ini: Multiplication factor for proposals
            (amounts to a initial covariance of (radius_ini ** 2) * cov_ini)
        radius_factor: contraction factor for the covariance when failing to find lower score.
        no_change_max: Number of samples drawn without finding a parameter achieving lower
            score before the covariance is contracted by radius_factor ** 2
        parallel: should the calls to the score be parallelized (during each step)
        vectorized: is the score function assumed to be vectorized? Default is False. If True,
            parallel is disregarded
        print_rec: specify how often should there be prints.
            Information on the optimisation is printed every print_rec steps if silent is False
        silent: should there be any prints?

    Outputs:
        An OptimResult object with attributes
            opti_param, opti_score, converged, hist_param, hist_scores.
    """
    # vectorized takes precendence on parallel
    if vectorized:
        parallel = False

    prop_radius = radius_ini
    # Interpret number of parameters to be drawn
    if per_step is None:
        if parallel:
            per_step = cpu_count()
        else:
            per_step = 1

    param_shape = param_ini.shape
    n_param = param_ini.size

    if prop_cov is None:
        prop_cov = np.eye(n_param)
    if prop_cov.shape[0] != n_param:
        raise Exception("Non conformant shape of correlation matrix and parameter")

    start_uncert_x = np.sqrt(np.max(np.diag(prop_cov)))

    start_param = param_ini
    start_score = score(start_param)

    accu_param = [start_param]
    accu_score = [start_score]

    proposals = np.random.multivariate_normal(
        mean=np.zeros(n_param), cov=prop_cov, size=chain_length * per_step
    )

    proposals = proposals.reshape((chain_length, per_step) + param_shape)

    no_change_count = 0
    n_change = 0

    if parallel:
        pool = Pool()  # pylint: disable=E1102

    # Main routine
    i = 0
    converged = False
    while (i < chain_length) & (not converged):
        prop_mod = proposals[i]

        if i > 30:
            avg_speed = (accu_score[i - 30] - accu_score[i]) / 30
        else:
            avg_speed = -1.0

        if i % print_rec == 0:
            blab(silent, f"Score at step {i}: {start_score}")
            if avg_speed >= 0:
                blab(
                    silent,
                    f"Average speed: {(accu_score[i - 30] - accu_score[i])/30} / step (averaged on 30 steps)",
                )

        prop_params = start_param + prop_radius * prop_mod

        if parallel:
            prop_scores = pool.map(score, prop_params)
        elif vectorized:
            prop_scores = score(prop_params)
        else:
            prop_scores = [score(prop_param) for prop_param in prop_params]
        prop_scores = np.array(prop_scores)
        arg_min = np.argmin(prop_scores)

        # Check if more than no_change_max successive draws have worse score
        lower_radius = np.argmin(prop_scores > start_score) >= no_change_max

        if prop_scores[arg_min] < start_score:
            start_param = prop_params[arg_min]
            start_score = prop_scores[arg_min]
            no_change_count = 0
            n_change += 1
        else:
            no_change_count += per_step

        accu_param.append(start_param)
        accu_score.append(start_score)

        lower_radius = (no_change_count >= no_change_max) or lower_radius
        if lower_radius:
            no_change_count = 0
            prop_radius = prop_radius * radius_factor

            blab(silent, f"New proposal radius: {prop_radius}")

        # Check termination
        i = i + 1

        converged_y = (avg_speed >= 0) & (avg_speed < ftol)
        converged_x = start_uncert_x * prop_radius < xtol
        converged = converged_y or converged_x

    if parallel:
        pool.close()

    if converged:
        if converged_x:
            msg = f" (updt on x < {xtol})"
        else:
            msg = f" (updt on f < {ftol})"
        print(f"Converged after {i} iterations" + msg)
    else:
        print("Optimisation algorithm did not converge")

    return OptimResult(
        opti_param=accu_param[-1],
        converged=converged,
        opti_score=accu_score[-1],
        hist_param=accu_param,
        hist_score=accu_score,
    )
