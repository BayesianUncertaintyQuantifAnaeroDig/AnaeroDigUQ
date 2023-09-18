from functools import partial
from typing import Optional, Type

import numpy as np

from ...bayes import AccuSampleVal, OptimResultVI, variational_inference
from .._typing import DigesterFeed, DigesterState, DigesterStates
from ..prediction_error import score_free_param
from ..proba import distr_param_map, ref_distr_param


def am2_vi(
    # Computation of score
    obs: DigesterStates,
    influent_state: DigesterFeed,
    initial_state: DigesterState,
    # Initial parameter and setting
    prior_param: np.ndarray = ref_distr_param,
    ini_post: Optional[np.ndarray] = None,
    temperature: float = 0.01,
    index_train: Optional[list[int]] = None,
    VI_method: str = "corr_weights",
    prev_eval: Optional[Type[AccuSampleVal]] = None,
    # Main arguments
    chain_length: int = 100,
    per_step: int = 100,
    per_step_eval: int = 10000,
    step_size: float = 0.01,
    xtol: float = 10 ** (-8),
    # Secondary argument
    k: Optional[int] = None,
    gen_decay: float = 0.2,
    momentum: float = 0.2,
    refuse_conf: float = 0.95,
    corr_eta: float = 0.7,
    parallel: bool = True,
    print_rec: int = 10,
    silent: bool = False,
    **kwargs
) -> OptimResultVI:
    r"""
    Perform Variational inference routine for AM2 modelisation

    The distribution map is fixed (gaussian in log-space).

    Optimises the following score:
        $$J(distr_param) = \int RMS (AM2(pred_param), obs) d pi(distr_param)(pred_param)
                            + KL(pi(distr_param), prior)$$

    over distributions mapped by distr_map.

    As usual in PyAM2, the RMS is on the log residuals.

    args:
        obs, influent_state, initial_state, digester_info, solver_method are passed to AM2
        prior_param, init_post are parameters defining respectively the prior distribution
            and the initial posterior distribution

    Further arguments are passed to variational_inference function from bayes module

    Note:
        The outputs currently generates samples as np.ndarray.

        Score is defined as am2_err(run_am2(x.to_dig_param(), ... ), obs )
        Then passed to variational_gauss_bayes routine from bayes module

    Try to minimize:
        J(post) = Score(post) + temperature *  KL(post, prior)

    Computes gradient at post = (mu, cov) through
    d_mu(Score) = \int Cov^{-1}(X- mu) score(X) post(dX)
    d_cov(Score) = \int .5 * ( Cov^{-1} (X-\mu) (Cov^{-1}(X-\mu))^T) score(X) post(dX)
    with score being centered for the posterior distr
    """

    score = partial(
        score_free_param,
        obs=obs,
        influent_state=influent_state,
        initial_state=initial_state,
        **kwargs
    )

    return variational_inference(
        fun=score,
        distr_map=distr_param_map,
        prior_param=prior_param,
        post_param=ini_post,
        temperature=temperature,
        prev_eval=prev_eval,
        index_train=index_train,
        VI_method=VI_method,
        eta=step_size,
        chain_length=chain_length,
        per_step=per_step,
        per_step_eval=per_step_eval,
        xtol=xtol,
        k=k,
        gen_decay=gen_decay,
        momentum=momentum,
        refuse_conf=refuse_conf,
        corr_eta=corr_eta,
        parallel=parallel,
        vectorized=False,  # This is not an option considering how the score is computed
        print_rec=print_rec,
        silent=silent,
    )
