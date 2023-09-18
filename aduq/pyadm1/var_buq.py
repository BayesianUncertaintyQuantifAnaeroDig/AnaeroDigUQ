"""
Bayesian inspired joint calibration and uncertainty quantification for ADM1
This algorithm is based on bayes_variational and iter_prior from adm1.optim module.
Both the prior and variational class are fixed for this function.

This function does not allow for much flexibility. Users interested in tweaking hyper parameters
should consider using functions from either adm1.optim module or bayes module.
"""

import numpy as np

from ..bayes import OptimResultVI
from .IO import DigesterFeed, DigesterInformation, DigesterState, DigesterStates
from .optim import adm1_iter_prior_vi, adm1_vi
from .proba import (
    distr_param_indexes,
    distr_param_map,
    ref_distr_param,
    ref_t_distr_param,
)


def adm1_var_bayes_uq(
    obs: DigesterStates,
    influent_state: DigesterFeed,
    initial_state: DigesterState,
    digester_info: DigesterInformation,
    # Problem definition
    temperature: float = 0.001,
    # Main arguments
    chain_length: int = 250,
    gen_decay: float = 0.2,
    momentum: float = 0.985,
    refuse_conf: float = 0.9,
    corr_eta: float = 0.7,
    # Main arguments
    **kwargs,
) -> OptimResultVI:
    r"""
    VarBug algorithm for ADM1

    The calibration algorithm is performed in two phases:
        - Roughly learning mean and standard deviation through iter_prior_vi
        - Learning block diagonal covariance through adm1_vi

    Optimises the following score:
        $$J(distr_param) = \int RMS (ADM1(pred_param), obs) d pi(distr_param)(pred_param)
                            + KL(pi(distr_param), prior)$$

    over gaussian distributions with block diagonal covariance on the log-parameters.

    As usual in pyadm1, the RMS is on the log residuals (see adm1_err documentation)

    args:
        obs, influent_state, initial_state, digester_info, solver_method are passed to ADM1
        temperature: the PAC-Bayesian temperature of the problem
    **kwargs are passed to adm1_err function

    output:
        OptimResultVI object with attribute opti_param.

    The gaussian parameter on the log parameter can be constructed through
    >>> from aduq.pyadm1.proba import distr_param_map
    >>> from aduq.pyadm1 import var_bayes_uq
    >>> out = var_bayes_uq(obs, influent_state, initial_state, digester_info, temperature)
    >>> posterior = distr_param_map(out.opti_param)


    Note:
        This function has little flexibility. Consider using adm1_iter_prior_vi and adm1_vi
            from pyadm1.optim module if hyperparameters need to be refined.
        This function will take a long time.
    """

    # First phase is iter_prior
    print("Starting first phase (iter_prior)\n")
    out = adm1_iter_prior_vi(
        obs=obs,
        influent_state=influent_state,
        initial_state=initial_state,
        digester_info=digester_info,
        prior_param=ref_t_distr_param,
        temperature=temperature,
        post_param=None,
        gen_per_step=512,
        chain_length=20,
        keep=126,
        frac_keep=0.25,
        frac_sparse=0.0,
        stop_tol=0.0,
        parallel=True,
        interactive=False,
        silent=False,
        **kwargs,
    )
    print("End of first phase\n")

    end_score = out.all_scores[0]
    print(f"Best score found: {end_score}")

    opti_param = out.opti_param
    # Convert from TensorizedGaussianMap param to GaussianMap param
    inter = np.zeros((opti_param.shape[1] + 1, opti_param.shape[1]))
    inter[0] = opti_param[0]
    inter[1:] = np.diag(opti_param[1])

    opti_param = inter

    print("Starting second phase (VI GD)\n")
    result = adm1_vi(
        obs=obs,
        influent_state=influent_state,
        initial_state=initial_state,
        digester_info=digester_info,
        distr_map=distr_param_map,  # fixed
        temperature=temperature,
        prev_eval=None,  # fixed
        prior_param=ref_distr_param,  # fixed
        post_param=opti_param,
        index_train=distr_param_indexes,  # fixed
        VI_method="corr_weights",  # fixed
        step_size=0.025,  # fixed
        chain_length=chain_length,
        per_step=160,
        xtol=10 ** (-8),
        k=None,
        gen_decay=gen_decay,
        momentum=momentum,
        refuse_conf=refuse_conf,
        corr_eta=corr_eta,
        parallel=True,
        print_rec=1,
        silent=False,
        **kwargs,
    )
    print("End of second phase\n")

    return result
