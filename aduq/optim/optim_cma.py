"""
CMA-ES algorithm.
See demo or https://doi.org/10.48550/arXiv.1604.00772 for description.

Implementation note:
    - par_eval is not used here since opening/closing pools of workers is unnecessary and takes
    time. This is not crucial though if scoring functions are intensive.
"""

from typing import Any, Callable, Optional, Type

import numpy as np
from multiprocess import Pool, cpu_count  # pylint: disable=E0611

from ..misc import ShapeError, blab
from .optim_result import OptimResultCMA


def optim_CMA(
    param_ini: Type[np.ndarray],
    score: Callable[[Any], float],
    # Termination criteria
    xtol: float = 10 ** (-8),
    ftol: float = 10 ** (-8),
    chain_length: int = 100,
    # Main arguments
    cov_ini: Optional[Type[np.ndarray]] = None,
    radius_ini: float = 1.0,
    # Further arguments
    per_step: int = 200,
    no_change_max: int = 10,
    radius_factor: Optional[float] = None,
    cov_updt_speed: float = 0.1,
    keep_frac: float = 0.25,
    n_average_speed: int = 30,
    parallel: bool = True,
    vectorized: bool = False,
    print_rec: int = 5,
    silent: bool = False,
) -> OptimResultCMA:
    """
    Optimisation algorithm using Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
    algorithm. CMA-ES algorithm is described in https://doi.org/10.48550/arXiv.1604.00772
    The implementation is original.

    Args:
        param_ini: initial mean parameter
        score: scoring function, to be minimized
        chain_length: maximum length of the chain
        xtol, ftol: criteria for termination (converged when one of the criteria is met)
        cov_ini: Initial covariance structure on parameters
        radius_ini: Multiplication factor for proposals
            (amounts to a initial covariance of (radius_ini ** 2) * cov_ini)
        per_step: Number of samples generated and evaluated at each step
        no_change_max: Number of samples drawn without finding a parameter achieving lower
            score before the covariance is contracted by radius_factor ** 2
        radius_factor: contraction factor for the covariance when failing to find lower score.
        cov_updt_speed: control on the speed of the covariance update. The covariance at time t+1 is
            (1- cov_updt_speed) * cov_t + cov_updt_speed * cov_updt .
            Default is 0.1
        keep_frac: fraction of good draws used to define the update of the covariance.
            Default is 0.25
        n_average_speed: number of steps used to compute the current average y decrease speed.
            Used for termination. Default is 30.
        parallel: should the calls to the score be parallelized (during each step)
        vectorized: is the score function assumed to be vectorized? Default is False. If True,
            parallel is disregarded
        print_rec: specify how often should there be prints.
            Information on the optimisation is printed every print_rec steps if silent is False
        silent: should there be any prints?

    Outputs:
        An OptimResultCMA object, inherited from OptimResults, with attributes
            opti_param, opti_score, converged, hist_param, hist_scores, hist_cov.
    """

    ## Set up defaults
    # vectorized takes precendence on parallel
    if vectorized:
        parallel = False

    # Choose radius_factor from cov_updt_speed if missing
    if radius_factor is None:
        radius_factor = np.sqrt(1 - cov_updt_speed)

    # Choose number of draws per optimisation step if not specified
    if per_step is None:
        if parallel:
            per_step = cpu_count()
        else:
            per_step = 1

    # Infer shape of parameters
    param_shape = param_ini.shape
    n_param = param_ini.size

    # Construct initial covariance
    if cov_ini is None:
        start_cov = (radius_ini**2) * np.eye(n_param)
    else:
        start_cov = (radius_ini**2) * cov_ini
    if start_cov.shape != (n_param, n_param):
        raise ShapeError(
            str.join(
                "\n",
                [
                    f"Covariance should be a square matrix of shape {(n_param, n_param)}",
                    f"Matrix shape is {cov_ini.shape}",
                ],
            )
        )

    # Set up accus
    start_param = param_ini.copy()
    start_score = score(start_param)

    accu_param = [start_param]
    accu_score = [start_score]
    accu_cov = [start_cov]

    max_keep = max(int(per_step * keep_frac), 1)

    count_fails = 0

    # Open pool once and for all for more efficiency
    if parallel:
        pool = Pool()  # pylint: disable=E1102

    # Main routine
    i = 0
    converged = False
    while (i < chain_length) & (not converged):
        # Analysis of optimisation procedure
        # Compute average speed
        if i > n_average_speed:
            avg_speed = (
                accu_score[i - n_average_speed] - accu_score[i]
            ) / n_average_speed
        else:
            avg_speed = -1.0

        # Prints
        if i % print_rec == 0:
            blab(silent, f"Score at step {i}: {start_score}")
            # If avg_speed is computed, give info
            if avg_speed >= 0:
                blab(
                    silent,
                    f"Average speed: {avg_speed} / step (averaged on {n_average_speed} steps)",
                )

        # Sample from current distribution
        draw = np.random.multivariate_normal(start_param, start_cov, per_step).reshape(
            (per_step,) + param_shape
        )

        # Evaluate score on sample
        if parallel:
            evals = np.array(pool.map(score, draw))
        elif vectorized:
            evals = score(draw)
        else:
            evals = np.array([score(param) for param in draw])

        # Count number of draws before finding a better score
        # If no better score founds, dealt through count_fail
        conseq_fails = np.argmin(evals >= start_score)

        # Sort scores
        sorter = np.argsort(evals)
        keep_draw = draw[sorter][:max_keep]
        evals = evals[sorter][:max_keep]

        if evals[0] <= start_score:
            # Case where a better score was found

            # The sample considered only parameters achieving a better score
            keep_draw = keep_draw[evals <= start_score]
            evals = evals[evals <= start_score]

            # The new center of the distribution is the parameter achieving the best score so far
            new_param = keep_draw[0]
            new_score = evals[0]

            # Compute soft covariance update
            # Compute pseudo covariance of sample, using the previous mean
            # This helps preferentially drawing along "new_param - start_param" axis
            add_cov = np.zeros(start_cov.shape)
            for param in keep_draw:
                add_cov = add_cov + np.outer(param - start_param, param - start_param)
            add_cov = add_cov / len(keep_draw)

            # Update current score, param and covariance
            start_score = new_score
            start_param = new_param
            start_cov = (1 - cov_updt_speed) * start_cov + (cov_updt_speed) * add_cov

            # Update the successive failure (i.e. no score decrease) count
            count_fails = 0
        else:
            # Update the successive failure (i.e. no score decrease) count
            count_fails += per_step

        # Check if covariance contraction is necessary (cumulative effect, or local effect)
        # Note that covariance contraction might happen even if the score decreased during
        # current step
        if max(conseq_fails, count_fails) >= no_change_max:
            blab(silent, "Updating covariance radius")
            start_cov = (radius_factor**2) * start_cov
            count_fails = 0

        # Add new parameter to accu
        accu_param.append(start_param)
        accu_score.append(start_score)
        accu_cov.append(start_cov)
        # Update step count
        i = i + 1

        # Check for convergence
        converged_y = (avg_speed >= 0) & (avg_speed < ftol)
        converged_x = np.sqrt(np.max(np.diag(start_cov))) < xtol
        converged = converged_x or converged_y

    # End of optimisation prints
    if converged:
        if converged_x:
            msg = f" (updt on x < {xtol})"
        else:
            msg = f" (updt on f < {ftol})"
        blab(silent, f"Converged after {i} iterations" + msg)
    else:
        blab(silent, "Optimisation algorithm did not converge")

    # Return OptimResult object
    return OptimResultCMA(
        opti_param=accu_param[-1],
        converged=converged,
        opti_score=accu_score[-1],
        hist_param=accu_param,
        hist_score=accu_score,
        hist_cov=accu_cov,
    )


class CMA_optimiser:
    """
    Class for CMA optimisation

    The optimisation task is set up, then optimisation is performed through the train method.
    This allows for more control on optimisation (multiple phases, change of optimisation meta parameters).
    """

    def __init__(self, par_ini=None, cov_ini=None, radius_ini=None, converged=False):
        self.param = par_ini
        self.cov = cov_ini
        if radius_ini is None:
            radius_ini = 1
        self.radius = radius_ini
        self.converged = converged

    def train(
        self,
        chain_length: int,
        score: Callable[[Any], float],
        per_step: int = 200,
        no_change_max=10,
        radius_factor=None,
        parallel: bool = True,
        vectorized: bool = False,
        print_rec: int = 5,
        cov_updt_speed: float = 0.1,
        keep_frac=0.25,
        silent=False,
    ):
        """
        Call to optim_CMA function for optimisation
        """

        res = optim_CMA(
            self.param,
            chain_length=chain_length,
            score=score,
            per_step=per_step,
            cov_ini=self.cov,
            radius_ini=self.radius,
            no_change_max=no_change_max,
            radius_factor=radius_factor,
            parallel=parallel,
            vectorized=vectorized,
            print_rec=print_rec,
            cov_updt_speed=cov_updt_speed,
            keep_frac=keep_frac,
            silent=silent,
        )
        blab(
            silent,
            "Training completed",
            f"New parameter: {res.opti_param[-1]}",
            f"Score went from {res.hist_score[0]} to {res.opti_score}",
            sep="\n",
        )
        self.param = res.opti_param
        self.cov = res.hist_cov[-1]
        self.converged = res.converged
