"""
Iter prior algorithm, inspired by A. Leurent and R. Moscoviz (https://doi.org/10.1002/bit.28156)

- OptimResultPriorIter subclass of OptimResult for iter_prior function
- iter_prior function
- iter_prior_vi, adaptation of iter_prior for Gaussian Variational problem
"""

import sys
from typing import Callable, List, Optional, Type, Union

import numpy as np

from ..misc import blab, par_eval
from ..optim import OptimResult
from ..proba import TensorizedGaussian, TensorizedGaussianMap


class OptimResultPriorIter(OptimResult):
    """Subclass of OptimResult for prior_iter algorithm"""

    def __init__(
        self,
        opti_param,
        opti_distrib: TensorizedGaussian,
        converged: bool,
        opti_score: Optional[float] = None,
        hist_param: Optional[list] = None,
        hist_score: Optional[List[float]] = None,
        full_sample: Optional[List] = None,
        track_gen: Optional[List[int]] = None,
        all_scores: Optional[List] = None,
    ):
        super().__init__(
            opti_param=opti_param,
            converged=converged,
            opti_score=opti_score,
            hist_param=hist_param,
            hist_score=hist_score,
        )
        self.opti_distrib = opti_distrib
        self.full_sample = full_sample
        self.track_gen = track_gen
        self.all_scores = all_scores


def construct_prior(sample: Type[np.ndarray], frac_sparse: int = 0) -> np.ndarray:
    """
    Construct a gaussian distribution from a sample.
    Covariance of constructed distribution is diagonal.

    The output is a ProbaParam numpy.ndarray which can be passed to appropriate
    TensorizedGaussianMap.

    Args:
        sample: a sample from a (sparsified) gaussian distribution.
        frac_sparse: fraction of features which are set to their means.
    """
    means = np.apply_along_axis(np.mean, 0, sample)
    devs = np.apply_along_axis(np.std, 0, sample) / np.sqrt(1 - frac_sparse)
    return np.array([means, devs])


def iter_prior(
    score_fun: Callable[[Type[np.ndarray]], float],
    ini_prior_param: np.ndarray,
    gen_per_step: Union[int, List[int]] = 100,
    chain_length: int = 1,
    keep: int = 250,
    frac_sparse: float = 0.0,
    parallel: bool = True,
    vectorized: bool = False,
    interactive: bool = True,
    silent: bool = True,
) -> OptimResultPriorIter:
    """
    Prior iteration joint Calibration and Uncertainty quantification routine.

    This code is adapted from a technique described in https://doi.org/10.1002/bit.28156

    Given a score function and a starting prior as a tensorized gaussian distribution,
    updates the prior by iteratively:
        - drawing samples from it,
        - fiting a new distribution from the k-best samples found so far

    The distribution is fitted by computing the mean and standard deviation
    of the best parameters, keeping "keep" of them. If less than keep samples
    where drawn so far, the prior is not updated.

    Args:
        score_fun: a scoring function
        ini_prior: the initial distribution parameter. Passed to a TensorizedGaussianDistrib object
            (fist element are the means, second element the standard deviation)
        gen_per_step: number of samples generated at each step
        chain_length: number of iterations
        keep: number of samples kept to prepare the next generation
        frac_sparse: fraction of features set to their mean
        parallel: should score_fun calls be parallelized? Default is True, suited to non
            negligeable computation time in score_fun
        interactive: should the print be interactive?
        silent: should there be any print?

    Outputs an OptimResultPriorIter object which is assumed not to have converged.
    """
    prior_param = np.array(ini_prior_param)
    out_shape = prior_param.shape[1:]

    distrib_map = TensorizedGaussianMap(sample_shape=out_shape)

    if isinstance(gen_per_step, int):
        gen_per_step = np.full(chain_length, gen_per_step)
    else:
        chain_length = len(gen_per_step)

    hist_param = np.zeros((chain_length, 2) + out_shape)
    hist_score = np.zeros(chain_length)

    tot_calls = np.sum(gen_per_step)
    accu_score = np.full(tot_calls, np.inf)
    accu_sample = np.zeros((tot_calls,) + out_shape)

    filled = 0
    m_score = np.inf

    # Set up sample generation tracker
    track_gen = []

    for i, n_sample in enumerate(gen_per_step):
        # Print information on last generation
        if interactive:
            sys.stdout.write("\r")
            sys.stdout.write(
                "Step %i/%i (mean score: %f)" % (i + 1, len(gen_per_step), m_score)
            )
            sys.stdout.flush()
        else:
            blab(
                silent,
                "Step %i/%i (mean score: %f)" % (i + 1, len(gen_per_step), m_score),
            )

        # Start loop properly
        prior = distrib_map(prior_param)
        sample = prior(n_sample)

        # Sparsify prior
        if frac_sparse > 0:
            for samp in sample:
                choose_k = np.random.choice(len(samp), int(frac_sparse * len(samp)))
                samp[choose_k] = prior.means[choose_k]

        if vectorized:
            scores = score_fun(sample)
        else:
            scores = par_eval(score_fun, sample, parallel=parallel)

        accu_sample[filled : (filled + n_sample)] = np.array(sample)
        accu_score[filled : (filled + n_sample)] = np.array(scores)
        track_gen.extend(np.full(n_sample, i))
        filled += n_sample

        sorter = accu_score[:filled].argsort()
        accu_score[:filled] = accu_score[sorter]
        accu_sample[:filled] = accu_sample[sorter]
        track_gen = [track_gen[i] for i in sorter]

        best_sample = accu_sample[: min(filled, keep)]
        m_score = np.mean(scores)

        if filled >= keep:
            prior_param = construct_prior(best_sample, frac_sparse)

        hist_param[i] = prior_param
        hist_score[i] = m_score
    blab(silent, "")

    return OptimResultPriorIter(
        opti_param=prior_param,
        opti_distrib=distrib_map(prior_param),
        converged=False,
        hist_param=hist_param,
        hist_score=hist_score,
        full_sample=accu_sample,
        track_gen=track_gen,
        all_scores=accu_score,
    )


def iter_prior_vi(
    score_fun: Callable[[Type[np.ndarray]], float],
    prior_param: np.ndarray,
    temperature: float = 0.0,
    post_param: Optional[np.ndarray] = None,
    gen_per_step: Union[int, List[int]] = 100,
    chain_length: int = 1,
    keep: int = 250,
    frac_sparse: float = 0.0,
    stop_tol: float = 0.0,
    parallel: bool = True,
    vectorized: bool = False,
    interactive: bool = True,
    silent: bool = True,
) -> OptimResultPriorIter:
    """
    Prior iteration joint Calibration and Uncertainty quantification routine.
    This is mainly thought as a jump-start procedure for variational inference methods

    Given a score function and a starting prior as a tensorized gaussian distribution,
    updates the prior by iteratively:
        - drawing samples from it,
        - fiting a new distribution from the k-best samples found so far
    Stops if the variational inference criteria stops decreasing.

    The distribution is fitted by computing the mean and standard deviation
    of the best parameters, keeping "keep" of them. If less than keep samples
    where drawn so far, the prior is not updated.

    Args:
        score_fun: scoring function
        prior_param: prior parameter for a tensorized gaussian distribution
        temperature: PAC-Bayesian temperature
        post_param: optional initial posterior parameter for a tensorized gaussian distribution
    Secondary arguments:
        gen_per_step: number of samples generated at each step
        chain_length: number of iterations
        keep: number of samples kept to prepare the next generation
        frac_sparse: fraction of features set to their mean
        parallel: should score_fun calls be parallelized? Default is True, suited to non
            negligeable computation time in score_fun
        interactive: should the print be interactive?
        silent: should there be any print?

    Outputs an OptimResultPriorIter object which is assumed not to have converged.
    """

    # Convert to np.array for safety's sake
    prior_param = np.array(prior_param)
    out_shape = prior_param.shape[1:]

    if post_param is None:
        post_param = prior_param.copy()
    else:
        post_param = np.array(post_param).copy()

    distrib_map = TensorizedGaussianMap(sample_shape=out_shape)

    if isinstance(gen_per_step, int):
        gen_per_step = np.full(chain_length, gen_per_step)
    else:
        chain_length = len(gen_per_step)

    hist_param = np.zeros((chain_length, 2) + out_shape)
    hist_score = np.zeros(chain_length)

    tot_calls = np.sum(gen_per_step)
    accu_score = np.full(tot_calls, np.inf)
    accu_sample = np.zeros((tot_calls,) + out_shape)

    filled = 0
    m_score = np.inf
    kl = 0

    # Set up sample generation tracker
    track_gen = []
    converged = False

    # tol = norm().ppf(stop_tol)
    V_prev = np.inf

    i = 0
    while (i < len(gen_per_step)) & (not converged):
        n_sample = gen_per_step[i]
        # Print information on last generation
        if interactive:
            sys.stdout.write("\r")
            sys.stdout.write(
                "Step %i/%i (mean score: %f, kl : %f)"
                % (i + 1, len(gen_per_step), m_score, kl)
            )
            sys.stdout.flush()
        else:
            blab(
                silent,
                "Step %i/%i (mean score: %f, kl: %f)"
                % (i + 1, len(gen_per_step), m_score, kl),
            )

        # Start loop properly
        kl = distrib_map.kl(post_param, prior_param)
        post = distrib_map(post_param)
        sample = post(n_sample)

        # Sparsify prior
        if frac_sparse > 0:
            for samp in sample:
                choose_k = np.random.choice(len(samp), int(frac_sparse * len(samp)))
                samp[choose_k] = post_param[0][choose_k]

        if vectorized:
            scores = score_fun(sample)
        else:
            scores = par_eval(score_fun, sample, parallel=parallel)

        accu_sample[filled : (filled + n_sample)] = np.array(sample)
        accu_score[filled : (filled + n_sample)] = np.array(scores)
        track_gen.extend(np.full(n_sample, i))
        filled += n_sample

        sorter = accu_score[:filled].argsort()
        accu_score[:filled] = accu_score[sorter]
        accu_sample[:filled] = accu_sample[sorter]
        track_gen = [track_gen[i] for i in sorter]

        best_sample = accu_sample[: min(filled, keep)]
        m_score = np.mean(scores)

        V = m_score + temperature * kl

        if V < V_prev + stop_tol:

            if filled >= keep:
                post_param = construct_prior(best_sample, frac_sparse)

            hist_param[i] = post_param
            hist_score[i] = m_score
            converged = False
            V_prev = V
        else:
            print(f"Converged (mean: {m_score}, kl: {kl})")
            converged = True

        i += 1
    blab(silent, "")

    return OptimResultPriorIter(
        opti_param=post_param,
        opti_distrib=distrib_map(prior_param),
        converged=False,
        hist_param=hist_param,
        hist_score=hist_score,
        full_sample=accu_sample,
        track_gen=track_gen,
        all_scores=accu_score,
    )
