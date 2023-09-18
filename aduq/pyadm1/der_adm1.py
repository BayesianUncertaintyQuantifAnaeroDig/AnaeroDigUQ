"""
Derivative of ADM1 with respect to the calibration parameters.

Used to compute Fisher's information matrix (FIM) for UQ and to compute sensitivity values in
local sensitivity analysis.
"""

from functools import partial
from typing import Optional

import numpy as np

from ..misc import interpretation, num_der, post_modif
from .IO import (DigesterFeed, DigesterInformation, DigesterParameter,
                 DigesterState, DigesterStates, parameter_dict)
from .run_adm1 import run_adm1


def adm1_derivative(
    param: DigesterParameter,
    influent_state: DigesterFeed,
    initial_state: DigesterState,
    digester_info: DigesterInformation,
    params_to_der: Optional[list] = None,
    log_adm1=True,
    # adm1_out kept for compatibility with prev standards
    adm1_out: Optional[DigesterStates] = None,  # pylint: disable=W0613
    rel_step: Optional[float] = 10**-7,
    max_step: float = 0.5 / (60 * 24),
    parallel=True,
    **kwargs,
) -> np.ndarray:
    """
    Compute the Jacobian of (log-)ADM1 (run_adm1 implementation) close to param for parameters
    dimension in params_to_der. Time output is dropped.
    Built on top of comp_3d_jacob which does the main numerical derivation.

    Jacobian is outputed as a np.ndarray of shape (P, T, K) where
        P is number of parameters in params_eval
        T is the number of time points
        K is the number of predictions type

    adm1_derivative is user friendly (use pandas input rather than numpy).
    Conversion is done inside function.

    To avoid computation issues, relatives increments are considered to the parameter when
    computing the derivative (i.e. we consider ADM1(param_j * (1 + epsilon)) - AMD1(param_j)).
    The difference is then corrected for the value of param_j so that the output gives absolute
    derivative.

    With FUN = ADM1 or log(ADM1) depending on log_adm1, computes the derivative by approximating:
    (FUN(param + param_i * epsilon)) - FUN(param))/(epsilon * param_i)

    Args:
        - param, the parameter at which the derivative of ADM1 is to be computed
        - digester_info, solver_method -> see run_adm1 doc
        - params_to_der, the list of parameter dimensions on which to compute the derivative.
            Default is None, which amounts to all parameters
        - log_adm1: if False, computes the derivative of ADM1, if True, computes the derivative of
            log(ADM1).
            True by default.
        - adm1_out: Optional output of ADM1 at param (even if log_adm1 is True). If false, not
            needed. Currently disregarded in all cases.
    kwargs are passed to run_adm1

    Returns:
        The jacobian in shape P, T, K (P number of parameters, T number of time points, K number of
            observations)

    Note:
        The default max_step argument of run_adm1 is set to half a minute. This is necessary to
        neutralize the effect of the max_step argument for small variation on parameter (spurious
        high frequency, small amplitude perturbations which impact the quality of the derivative).
        For similar reasons, one ought to be careful not to set the rel_step too low (or force
        the max_step to be low as well).
    """

    # Prepare ADM1 function
    set_adm1 = partial(
        run_adm1,
        influent_state=influent_state,
        initial_state=initial_state,
        digester_info=digester_info,
        max_step=max_step,
        **kwargs,
    )

    if log_adm1:
        post_mod = post_modif(lambda y: np.log(y[:, 1:]))
    else:
        post_mod = post_modif(lambda y: y[:, 1:])

    if params_to_der is not None:
        params_index = [parameter_dict[name] for name in params_to_der]
        ref_param = np.array(param).copy()

        def to_param(x):
            new_par = ref_param.copy()
            new_par[params_index] = x
            return new_par

        translate_par = interpretation(to_param)

        # Reduce parameter to the values on which to compute derivatives
        ini_param = param[params_index].copy()
    else:
        params_to_der = list(parameter_dict.keys())
        translate_par = interpretation(lambda x: x)
        ini_param = param

    # Modify ADM1
    loc_adm1 = translate_par(post_mod(set_adm1))

    der = num_der(fun=loc_adm1, x0=ini_param, rel_step=rel_step, parallel=parallel)

    return der
