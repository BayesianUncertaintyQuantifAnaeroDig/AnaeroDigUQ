"""
Bayesian inspired joint calibration and uncertainty quantification for AM2.
This algorithm is based on bayes_variational and iter_prior from adm1.optim module.
Both the prior and variational class are fixed for this function.

This function does not allow for much flexibility. Users interested in tweaking hyper parameters
should consider using functions from either am2.optim module or bayes module.

This function is basically am2.optim.am2_variational_inference with options removed and different
defaults. The main difference is that the covariance learnt is by default block diagonal (no such
constraint is applied by default in am2_variational_inference).
"""

from typing import Optional

import numpy as np

from ..bayes import OptimResultVI
from ._typing import DigesterFeed, DigesterState, DigesterStates
from .optim import am2_vi
from .proba import distr_param_indexes, distr_param_map, ref_distr_param


def am2_var_bayes_uq(
    # Computation of score
    obs: DigesterStates,
    influent_state: DigesterFeed,
    initial_state: DigesterState,
    # Initial parameter and setting
    prior_param: np.ndarray = ref_distr_param,
    ini_post: Optional[np.ndarray] = None,
    temperature: float = 0.002,
    # Main arguments
    chain_length: int = 250,
    per_step: int = 256,
    step_size: float = 0.04,
    # Secondary argument
    gen_decay: float = 0.15,
    momentum: float = 0.96,
    refuse_conf: float = 0.95,
    corr_eta: float = 0.7,
    **kwargs
) -> OptimResultVI:
    r"""
    VarBug algorithm for ADM1

    The calibration algorithm is performed in by learning means and block diagonal covariance
    through am2_vi

    Optimises the following score:
        $$J(distr_param) = \int RMS (AM2(pred_param), obs) d pi(distr_param)(pred_param)
                            + KL(pi(distr_param), prior)$$

    over gaussian distributions with block diagonal covariance on the log-parameters.

    As usual in pyam2, the RMS is on the log residuals (see am2_err documentation)

    args:
        obs, influent_state, initial_state, solver_method are passed to run_am2
        temperature: the PAC-Bayesian temperature of the problem
    **kwargs are passed to adm1_err function

    output:
        OptimResultVI object with attribute opti_param.

    The gaussian parameter on the log parameter can be constructed through
    >>> from aduq.pyam2.proba import distr_param_map
    >>> from aduq.pyam2 import var_bayes_uq
    >>> out = var_bayes_uq(obs, influent_state, initial_state, temperature)
    >>> posterior = distr_param_map(out.opti_param)


    Note:
        This function calls am2_vi function and does nothing else.
        This function has little flexibility. Consider using am2_iter_prior_vi and am2_vi
            from pyam2.optim module if hyperparameters need to be refined.
        This function will take a long time.
    """

    return am2_vi(
        obs=obs,
        influent_state=influent_state,
        initial_state=initial_state,
        distr_map=distr_param_map,
        prior_param=prior_param,
        post_param=ini_post,
        temperature=temperature,
        index_train=distr_param_indexes,
        VI_method="corr_weights",
        step_size=step_size,
        chain_length=chain_length,
        per_step=per_step,
        gen_decay=gen_decay,
        momentum=momentum,
        refuse_conf=refuse_conf,
        corr_eta=corr_eta,
        parallel=True,
        print_rec=1,
        silent=False,
        **kwargs
    )
