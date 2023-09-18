""" Iter prior procedure for ADM1 model """
from functools import partial
from typing import Optional, Union

import numpy as np

from ...bayes.iter_prior import OptimResultPriorIter, iter_prior, iter_prior_vi
from .._typing import DigesterFeed, DigesterState, DigesterStates
from ..prediction_error import score_free_param
from ..proba import ref_t_distr_param


def am2_iter_prior(
    obs: DigesterStates,
    influent_state: DigesterFeed,
    initial_state: DigesterState,
    ini_prior_param: np.ndarray = ref_t_distr_param,
    gen_per_step: Union[int, list] = 100,
    chain_length: int = 10,
    keep: int = 250,
    interactive: bool = False,
    frac_sparse: float = 0,
    parallel: bool = True,
    silent: bool = False,
    **kwargs
) -> OptimResultPriorIter:
    """
    Wrapper of iter_prior routine to use with ADM1.
    It is assumed that the initial prior is defined in the FreeDigesterParameter space,
    but outputs numpy.ndarray sample (behavior if it outputs FreeDigesterParameter is not clear).
    The distribution outputed is modified so that it generates samples as a list of FreeDigesterParameter.

    """
    print("Starting prior iteration routine for AM2 calibration")
    score_fun = partial(
        score_free_param,
        obs=obs,
        influent_state=influent_state,
        initial_state=initial_state,
        silent=True,
        **kwargs
    )

    out = iter_prior(
        score_fun=score_fun,
        ini_prior_param=ini_prior_param,
        gen_per_step=gen_per_step,
        chain_length=chain_length,
        keep=keep,
        frac_sparse=frac_sparse,
        parallel=parallel,
        interactive=interactive,
        silent=silent,
    )

    return out


def am2_iter_prior_vi(
    obs: DigesterStates,
    influent_state: DigesterFeed,
    initial_state: DigesterState,
    prior_param: np.ndarray = ref_t_distr_param,
    temperature: float = 0.0,
    post_param: Optional[np.ndarray] = None,
    gen_per_step: Union[int, list] = 100,
    chain_length: int = 10,
    keep: int = 250,
    interactive: bool = False,
    frac_sparse: float = 0,
    stop_tol: float = 0.0,
    parallel: bool = True,
    silent: bool = False,
    **kwargs
) -> OptimResultPriorIter:
    """
    Wrapper of iter_prior routine to use with ADM1.
    It is assumed that the initial prior is defined in the FreeDigesterParameter space,
    but outputs numpy.ndarray sample (behavior if it outputs FreeDigesterParameter is not clear).
    The distribution outputed is modified so that it generates samples as a list of FreeDigesterParameter.

    """
    print("Starting prior iteration routine for AM2 calibration")
    score_fun = partial(
        score_free_param,
        obs=obs,
        influent_state=influent_state,
        initial_state=initial_state,
        silent=True,
        **kwargs
    )

    out = iter_prior_vi(
        score_fun=score_fun,
        prior_param=prior_param,
        temperature=temperature,
        post_param=post_param,
        gen_per_step=gen_per_step,
        chain_length=chain_length,
        keep=keep,
        frac_sparse=frac_sparse,
        stop_tol=stop_tol,
        parallel=parallel,
        vectorized=False,  # This is not an option considering how the score is computed
        interactive=interactive,
        silent=silent,
    )

    return out
