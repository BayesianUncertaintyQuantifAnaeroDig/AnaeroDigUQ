""" Iter prior procedure for ADM1 model """
import warnings
from functools import partial
from typing import Optional, Union

import numpy as np

from ...bayes.iter_prior import OptimResultPriorIter, iter_prior, iter_prior_vi
from ..IO import DigesterFeed, DigesterInformation, DigesterState, DigesterStates
from ..prediction_error import score_free_param
from ..proba import ref_t_distr_param


def adm1_iter_prior(
    obs: DigesterStates,
    influent_state: DigesterFeed,
    initial_state: DigesterState,
    digester_info: DigesterInformation,
    ini_prior_param: np.ndarray = ref_t_distr_param,
    gen_per_step: Union[int, list] = 100,
    chain_length: int = 1,
    keep: int = 250,
    frac_sparse: float = 0,
    parallel: bool = True,
    interactive: bool = False,
    silent: bool = False,
    **kwargs
) -> OptimResultPriorIter:
    """
    Wrapper of iter_prior routine to use with ADM1.
    It is assumed that the initial prior is defined in the FreeDigesterParameter space,
    but outputs numpy.ndarray sample (behavior if it outputs FreeDigesterParameter is not clear).

    Outputs an OptimResultPriorIter
    """
    score = partial(
        score_free_param,
        obs=obs,
        influent_state=influent_state,
        initial_state=initial_state,
        digester_info=digester_info,
        **kwargs
    )
    out = iter_prior(
        score_fun=score,
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


def adm1_iter_prior_vi(
    obs: DigesterStates,
    influent_state: DigesterFeed,
    initial_state: DigesterState,
    digester_info: DigesterInformation,
    prior_param: np.ndarray = ref_t_distr_param,
    temperature: float = 0.0,
    post_param: Optional[np.ndarray] = None,
    gen_per_step: Union[int, list] = 100,
    chain_length: int = 1,
    keep: Optional[int] = None,
    frac_keep: Optional[float] = 0.25,
    frac_sparse: float = 0,
    stop_tol: float = 0.0,
    parallel: bool = True,
    interactive: bool = False,
    silent: bool = False,
    **kwargs
) -> OptimResultPriorIter:
    """
    Wrapper of iter_prior routine to use with ADM1.
    It is assumed that the initial prior is defined in the FreeDigesterParameter space,
    but outputs numpy.ndarray sample (behavior if it outputs FreeDigesterParameter is not clear).

    Outputs an OptimResultPriorIter
    """
    if keep is None:
        keep = int(frac_keep * gen_per_step)

    if keep < 30:
        warnings.warn(
            "Insufficient number of samples used to estimate Tensorized Gaussian distribution. Raised to 30."
        )
        keep = 30

    score = partial(
        score_free_param,
        obs=obs,
        influent_state=influent_state,
        initial_state=initial_state,
        digester_info=digester_info,
        **kwargs
    )
    out = iter_prior_vi(
        score_fun=score,
        prior_param=prior_param.copy(),
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
