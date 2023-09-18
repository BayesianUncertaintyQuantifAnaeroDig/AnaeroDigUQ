"""
Uncertainty quantification using Beale's method
"""

import warnings
from functools import partial
from typing import Any, Callable, Optional, Union

import numpy as np
from scipy.stats import f

from ..misc import ShapeError, blab, par_eval, timeout


class NegativeCov(Warning):
    """Custom warning when encountering a covariance with some negative eigenvalues"""


class IncompleteSample(Warning):
    """Custom warning for Beale's boundary computation, informing that not all line search
    procedures succeeded"""


def find_boundary(
    input_data,
    opti_param: Union[float, np.ndarray],
    j_crit: float,
    score_func: Callable[[Any], dict],
    max_iter: int,
    relat_crit: float,
    max_time: int = 600,
    silent: bool = True,
) -> Optional[Union[float, np.ndarray]]:
    """
    Newton solver for F(l) := score_func(theta + l dir) = j_crit

    Errors arising from J computation will output None

    Timeout functionality (default is 10 minutes).

    Args:
        input_data: tuple containing id (for printing purposes)
            and direction (direction on which to conduct line search)
        opti_param: origin for line search
        j_crit: target score value
        score_func: scoring function
        max_iter: maximum number of iterations
        relat_crit: solver precision (relative: stop for abs(score/Jcrit -1) < relat_crit)
        max_time: maximum search time (in case scoring time can be arbitrarily low)
        silent: should there be no print? Default is False.

    Note:
        This function should never fail. Exceptions are caught and result in None output.
    """

    iter_id, direction = input_data
    blab(silent, f"Starting iteration {iter_id + 1}")

    with timeout(max_time):
        try:
            par = direction + opti_param
            j_curr = score_func(par)
            # Define a random step just by
            if j_curr < j_crit:
                delta = 0.05
            else:
                delta = -0.05

            j_old = j_curr
            par = par + delta * direction
            j_curr = score_func(par)
            rel_chi = np.abs(j_curr - j_crit) / j_crit

            curr_step = 0
            while (rel_chi > relat_crit) & (curr_step < max_iter):
                slope = (j_curr - j_old) / delta  ## Evaluate line derivative
                delta = (
                    j_crit - j_curr
                ) / slope  ## Newton guess value of par which values j_crit
                j_old = j_curr

                par = par + delta * direction  ## Moves to Newton guessed value
                j_curr = score_func(par)

                rel_chi = np.abs(j_curr - j_crit) / j_crit
                curr_step = curr_step + 1
        except TimeoutError:
            blab(silent, "Step failed due to timeout")
            curr_step = max_iter + 1
            rel_chi = relat_crit + 1
        except Exception as exc:  # pylint: disable=W0703
            blab(silent, "Step failed:", exc)
            curr_step = max_iter + 1
            rel_chi = relat_crit + 1

    if (curr_step < max_iter) or (rel_chi <= relat_crit):
        blab(silent, "Iteration succeeded")
        blab(silent, f"Parameter in log space: {np.array(par)}")
        return par

    return None


def beale_boundary(
    opti_param: np.ndarray,
    score_func: Callable[[np.ndarray], dict],
    cov: Optional[np.ndarray] = None,
    n_obs: int = 0,
    conf_lev: float = 0.95,
    n_boundary: int = 300,
    max_iter: int = 20,
    max_time: int = 600,
    relat_crit: float = 0.0001,
    parallel: bool = True,
    silent: bool = True,
) -> dict:
    """
    Estimate the limit of confidence region with confidence level conf_lev for parameters
    using Beale's method. For more documentation, see Dochain Vanrolleghem 2002.
    Adapted from D. Batstone's matlab code.

    For a scoring function score_func and a parameter opti_param,
    find the boundary of the set of parameters achieving
        score_func(param) / score_func(opti_param) < threshold
    by solving multiple line search problems of

        S(lambda) := score_func(opti_param + lambda * dir) = threshold * score_func(opti_param).

    The direction for each line search problem is drawn at random using the covariance matrix cov.

    Args:
        - opti_param gives the optimal parameter in array form.
        - pred_func is a function of parameter, working with arrays. pred_func should be able to
        work for all values of R^d (no constraints).
        - n_obs specifies the number of observations (to compute the threshold)
        - cov is a 2D array form of a covariance matrix, giving the first estimation
        of the confidence region.
        - max_iter is here to speed up computations
        (if you can not find boundary in the line between a param and opti_param in less than
        max_iter, drop it)
        - conf_lev is the confidence level for the confidence region to be estimated
        - n_boundary is the number of points in the confidence region boundary.
        - max_time: maximum evaluation time (in seconds) of score_func (stops the line search if
            time is exceeded)
        - relat_crit: precision used before stopping each line search. Automatically lowered if
            necessary
        - parallel: should the line search problems be parallelized?
        - silent: shoud there be regular prints?

    Output:
        - a Dictionnary with the following keys:
            "boundary": a numpy.ndarray containing the boundary
            "score_func": the scoring function used to compute the boundary
            "min_score": The minimal score (i.e. the score of opti_param)
            "conf_lev": The confidence level specified
            "n_obs": the number of observations
            "n_params": The number of parameters

    Remark:
        - If score_func raises an error, then the line search is stopped (but the procedure
            continues)
        - If score_func's evaluation time is too long (more than max_time), then the line search is
            stopped
        - The relat_crit parameter is adjusted so that the threshold Score_t is met with precision
            at least (Score_t - Score_min) * 0.05
            This prevents situations where the initial point, achieving score_min, would be
            considered to be a sufficiently good approximation of the boundary!

    Potential improvement:
        - loop until enough values have been found

    Methodological issues
        - Sizes badly with dimension of uncertainty region. Impractical whenever d > 3
        - Does not work for constrained parameters.
    """

    ## Set up inputs
    # Set covariance matrix to Id if no previous information
    n_params = len(opti_param)

    # Check if covariance exists/creates if not & checks format.
    if cov is None:
        cov = np.eye(n_params)
    elif cov.shape != (n_params, n_params):
        raise ShapeError(f"Covariance should have shape {(n_params, n_params)}")

    # Check that covariance is positive semi-definite (numerical errors can arise)
    cov = 0.5 * (cov + cov.T)  # Force symmetry

    eigs_covar = np.linalg.eigvalsh(cov)
    if eigs_covar[0] <= 0:
        count_bad = np.sum(eigs_covar <= 0.0)
        warnings.warn(
            f"Covariance matrix had {count_bad} negative eigenvalue(s).\nLowest eigenvalue is raised to 10 ** (-8)",
            category=NegativeCov,
        )
        cov = cov + (10 ** (-8) - eigs_covar[0]) * np.identity(n_params)

    # Evaluate optimal score
    j_min = score_func(opti_param)

    # Define threshold score
    ratio = 1 + (n_params / (n_obs - n_params)) * f.ppf(
        conf_lev, n_params, n_obs - n_params
    )
    j_crit = j_min * ratio

    # Print score information
    blab(
        silent,
        f"Threshold score is: {j_crit}, while min score is: {j_min}.\nThe ratio is {ratio}",
    )

    # Set up precision of line search procedure
    relat_crit_max = ratio - 1.0
    if relat_crit_max < 20 * relat_crit:
        warnings.warn(
            f"The criteria for finding the boundary is too lax. Setting it to {relat_crit_max * .05}"
        )
        relat_crit = relat_crit_max * 0.05

    # Prepare random directions
    parvecs = np.random.multivariate_normal(
        np.zeros(n_params), cov=cov, size=n_boundary
    )

    # Set up local version of find_boundary
    loc_find_boundary = partial(
        find_boundary,
        opti_param=opti_param,
        j_crit=j_crit,
        score_func=score_func,
        max_iter=max_iter,
        relat_crit=relat_crit,
        max_time=max_time,
        silent=silent,
    )

    # Main computation - Solving the line search problems
    parout = par_eval(loc_find_boundary, enumerate(parvecs), parallel)

    # Remove failed line search outputs
    parout = [x for x in parout if x is not None]
    n_failed = n_boundary - len(parout)

    # Warn if any failures
    if n_failed > 0:
        warnings.warn(f"{n_failed}/{n_boundary} line searches procedure failed.")
    else:
        blab(silent, "All line searches procedure succeeded.")

    return {
        "boundary": parout,
        "min_score": j_min,
        "n_params": n_params,
    }


def beale_pval(n_param: int, n_obs: int, score_param: float, score_opt: float):
    """
    Compute the p-value for the hypothesis that the observations were generated from a specific
    parameter under the model:

        obs = f(param) + noise

    assuming that the noise is gaussian and f is linear, using an F statistic.

    The p-value is computing using f-test, from the Mean squared prediction error of the specified
    parameter and the minimal mean squared prediction error.

    Args:
        n_param: the number of parameters fitted
        n_obs: the number of data points used for calibration
        score_param: the score of the param for which to compute the p-value (test whether
            the observations were generated from param)
        score_opt: minimal score found after optimisation

    Hypothesis:
        score is the Mean squared prediction error
        the statistical model uses additive gaussian noise on the predictions
        the predictions behave linearly with the parameters

    This last hypothesis is rarely assumed in practice.

    Justification:
        If the noise is gaussian and f linear, it follows that
        score_opt and score_param - score_opt are 2 independant chi2 distributed random variables
        This implies that (n_obs - n_param)/ n_param *  (score_param - score_opt)/ score_opt
        follows an F distribution.
    """
    ratio = score_param / score_opt

    val = (ratio - 1) * (n_obs - n_param) / (n_param)
    return 1 - f.cdf(val, n_param, n_obs - n_param)
