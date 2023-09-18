from functools import partial
from typing import Optional, Type

import numpy as np

from ...bayes import AccuSampleVal, OptimResultVI, variational_inference
from ...proba import ProbaMap
from ..IO import DigesterFeed, DigesterInformation, DigesterState, DigesterStates
from ..prediction_error import score_free_param
from ..proba import distr_param_map, ref_distr_param


def adm1_vi(
    obs: DigesterStates,
    influent_state: DigesterFeed,
    initial_state: DigesterState,
    digester_info: DigesterInformation,
    # Initial parameter and setting
    prior_param: np.ndarray = ref_distr_param,
    ini_post: Optional[np.ndarray] = None,
    distr_map: ProbaMap = distr_param_map,
    temperature: float = 0.001,
    VI_method: str = "corr_weights",
    prev_eval: Optional[Type[AccuSampleVal]] = None,
    # Main arguments
    chain_length: int = 250,
    per_step: int = 160,
    per_step_eval: int = 100000,  # For KNN VI method
    step_size: float = 0.025,
    xtol: float = 10 ** (-8),
    # Secondary argument
    index_train: Optional[list[int]] = None,
    k: Optional[int] = None,
    gen_decay: float = 0.2,
    refuse_conf: float = 0.9,
    corr_eta: float = 0.7,
    #    gen_weights: Optional[List] = None,
    momentum: float = 0.985,
    parallel: bool = True,
    print_rec: int = 10,
    silent: bool = False,
    **kwargs,
) -> OptimResultVI:
    r"""
    Perform a variational inference routine for ADM1 modelisation

    Optimises the following score:
        $$J(distr_param) = \int RMS (ADM1(pred_param), obs) d pi(distr_param)(pred_param)
                            + KL(pi(distr_param), prior)$$

    over distributions mapped by distr_map.

    As usual in PyADM1, the RMS is on the log residuals (see adm1_err documentation)

    args:
        obs, influent_state, initial_state, digester_info, solver_method are passed to ADM1
        prior_param, init_post are parameters defining respectively the prior distribution
            and the initial posterior distribution
        distr_map: the parametric family of distributions considered for optimization

    Note:
        The outputs currently generates samples as np.ndarray.

        Score is defined as adm1_err(run_adm1(x.to_dig_param(), ... ), obs )
        Then passed to variational_gauss_bayes routine from bayes module

    Try to minimize:
        J(post) = Score(post) + temperature *  KL(post, prior)

    Computes gradient at post = (mu, cov) through
        $$d_{mu}(Score) = \int Cov^{-1}(X- mu) score(X) post(dX)$$
        $$d_{cov}(Score) = \int .5 * ( Cov^{-1} (X-\mu) (Cov^{-1}(X-\mu))^T) score(X) post(dX)$$
    with score being centered for the posterior distr

    For information on routine, see documentation in bayes module.
    """

    score = partial(
        score_free_param,
        obs=obs,
        influent_state=influent_state,
        initial_state=initial_state,
        digester_info=digester_info,
        silent=True,
        **kwargs,
    )

    result = variational_inference(
        fun=score,
        distr_map=distr_map,
        temperature=temperature,
        prev_eval=prev_eval,
        prior_param=prior_param,
        post_param=ini_post,
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

    return result
