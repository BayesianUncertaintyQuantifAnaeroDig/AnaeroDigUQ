"""
Derivative of AM2 with respect to the calibration parameters.

Used to compute Fisher's information matrix (FIM) for UQ.
"""
import warnings
from functools import partial
from typing import Optional

import numpy as np
import scipy.integrate as si

from ..misc import interpretation, num_der, post_modif
from ._typing import DigesterFeed, DigesterParameter, DigesterState, DigesterStates
from .IO import param_to_pd, parameter_dict
from .run_am2 import (
    KH,
    P_T,
    ParamHandling,
    alpha,
    day_index,
    k1,
    k2,
    k3,
    k4,
    k5,
    k6,
    kLa,
    pH_col,
    pKb,
    run_am2,
)


def am2_with_der(
    param: DigesterParameter,
    influent_state: DigesterFeed,
    initial_state: DigesterState,
    solver_method: str = "LSODA",
    min_step: float = 10**-4,
    **kwargs,
) -> DigesterStates:

    # Read time
    ts = influent_state[:, 0]
    keep_index = day_index(ts)

    # Read parameter
    mu1max, mu2max, KS1, KS2, KI2 = param

    # Find max step_size
    if "max_step" not in kwargs.keys():
        # At least an evaluation every quarter of an hour for stability
        max_step = max(min(np.min(ts[1:] - ts[:-1]), 1 / 96), 20 / (96 * 60))
        kwargs["max_step"] = max_step

    # Check max_step > min_step
    if kwargs["max_step"] < min_step:
        warnings.warn(
            "Minimal step larger than maximum step. Setting min step to .5 * max step",
            category=ParamHandling,
        )
        min_step = kwargs["max_step"] / 2

    # Derivative mechanism
    def am2_ode(t: float, y: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the digester state (S) at time t, i.e.

        dS/dt (t) = AM2_ODE(t, S)
        """

        # Read feed information
        index = min(np.sum(t > ts), influent_state.shape[0] - 1)
        D, S1_in, S2_in, Z_in, C_in, pH = influent_state[index, 1:]

        # Unpack current digester state information
        X1, X2, S1, S2, Z, C = y[:6]

        # Compute intermediary
        mu1 = mu1max * (S1 / (KS1 + S1) - 0.1)  # D-1
        mu2 = mu2max * (S2 / (KS2 + S2 * (1 + S2 / KI2)) - 0.1)  # D-1

        d_mu1_mu1max = S1 / (KS1 + S1) - 0.1
        d_mu1_KS1 = mu1max * (-S1 / (KS1 + S1) ** 2)

        d_mu2_mu2max = S2 / (KS2 + S2 * (1 + S2 / KI2)) - 0.1
        d_mu2_KS2 = -mu2max * S2 / (KS2 + S2 * (1 + S2 / KI2)) ** 2
        d_mu2_KI2 = mu2max * (S2**3 / (KI2 * (KS2 + S2) + S2**2) ** 2)

        d_mu1_S1 = mu1max * (KS1 / (KS1 + S1) ** 2)
        d_mu2_S2 = mu2max * (
            1 / (KS2 + S2 * (1 + S2 / KI2))
            - S2 * (1 + 2 * S2 / KI2) / (KS2 + S2 * (1 + S2 / KI2)) ** 2
        )

        qM = k6 * mu2 * X2  # mmol L-1 D-1

        d_qM_mu2 = k6 * X2

        d_qM_X2 = k6 * mu2
        d_qM_S2 = k6 * d_mu2_S2 * X2

        CO2 = C / (1 + 10 ** (pH - pKb))
        d_CO2_C = 1 / (1 + 10 ** (pH - pKb))

        # CO2 = C + S2 - Z # mmol L-1

        phi = CO2 + KH * P_T + qM / kLa  # mmol L-1
        d_phi_C = d_CO2_C
        d_phi_qM = 1 / kLa

        KH_PC = (phi - np.sqrt(phi**2 - 4 * KH * P_T * CO2)) / 2  # mmol L-1
        d_KHPC_phi = (1 - phi / np.sqrt(phi**2 - 4 * KH * P_T * CO2)) / 2
        d_KHPC_CO2 = KH * P_T / (np.sqrt(phi**2 - 4 * KH * P_T * CO2))

        d_KHPC_C = d_KHPC_phi * d_phi_C + d_KHPC_CO2 * d_CO2_C
        d_KHPC_X2 = d_KHPC_phi * d_phi_qM * d_qM_X2
        d_KHPC_S2 = d_KHPC_phi * d_phi_qM * d_qM_S2

        qC = kLa * (CO2 - KH_PC)  # mmol L-1 D-1

        d_qC_mu2 = -kLa * d_KHPC_phi * d_phi_qM * d_qM_mu2

        d_qC_C = kLa * (d_CO2_C - d_KHPC_C)
        d_qC_X2 = -kLa * d_KHPC_X2
        d_qC_S2 = -kLa * d_KHPC_S2

        der_state_t = [
            (mu1 - alpha * D) * X1,  # dX1 # g L-1 D-1
            (mu2 - alpha * D) * X2,  # dX2 # mmol L-1 D-1
            D * (S1_in - S1) - k1 * mu1 * X1,  # dS1 # gCod L-1 D-1
            D * (S2_in - S2) + k2 * mu1 * X1 - k3 * mu2 * X2,  # dS2 # mmol L-1 D-1
            D * (Z_in - Z),  # dZ # mmol L-1 D-1
            D * (C_in - C) - qC + k4 * mu1 * X1 + k5 * mu2 * X2,  # dC # mmol L-1 D-1
        ]

        der_state_t_param = np.array(
            [
                [
                    X1 * d_mu1_mu1max,
                    0,
                    X1 * d_mu1_KS1,
                    0,
                    0,
                ],  # mu1max, mu2max, KS1, KS2, KI2
                [0, X2 * d_mu2_mu2max, 0, X2 * d_mu2_KS2, X2 * d_mu2_KI2],
                [-k1 * d_mu1_mu1max * X1, 0, -k1 * d_mu1_KS1 * X1, 0, 0],
                [
                    k2 * d_mu1_mu1max * X1,
                    -k3 * d_mu2_mu2max * X2,
                    k2 * d_mu1_KS1 * X1,
                    -k3 * d_mu2_KS2 * X2,
                    -k3 * d_mu2_KI2 * X2,
                ],
                [0, 0, 0, 0, 0],
                [
                    k4 * d_mu1_mu1max * X1,
                    (k5 * X2 - d_qC_mu2) * d_mu2_mu2max,
                    k4 * d_mu1_KS1 * X1,
                    (k5 * X2 - d_qC_mu2) * d_mu2_KS2,
                    (k5 * X2 - d_qC_mu2) * d_mu2_KI2,
                ],
            ]
        ).flatten()

        der_state_t_state = np.array(
            [
                [
                    mu1 - alpha * D,  # X1
                    0,  # X2
                    d_mu1_S1 * X1,  # S1
                    0,  # S2
                    0,  # Z
                    0,  # C
                ],  # dX1/dt
                [
                    0,  # X1
                    (mu2 - alpha * D),  # X2
                    0,  # S1
                    d_mu2_S2 * X2,  # S2
                    0,  # Z
                    0,  # C
                ],  # dX2/dt
                [
                    -k1 * mu1,  # X1
                    0,  # X2
                    -D - k1 * d_mu1_S1 * X1,  # S1
                    0,  # S2
                    0,  # Z
                    0,  # C
                ],  # dS1/dt
                [
                    k2 * mu1,  # X1
                    -k3 * mu2,  # X2
                    k2 * d_mu1_S1 * X1,  # S1
                    -D - k3 * d_mu2_S2 * X2,  # S2
                    0,  # Z
                    0,  # C
                ],  # dS2/dt
                [
                    0,  # X1
                    0,  # X2
                    0,  # S1
                    0,  # S2
                    -D,  # Z
                    0,  # C
                ],  # dZ/dt
                [
                    k4 * mu1,  # X1
                    k5 * mu2 - d_qC_X2,  # X2
                    k4 * d_mu1_S1 * X1,  # S1
                    k5 * d_mu2_S2 * X2 - d_qC_S2,  # S2
                    0,  # Z
                    -D - d_qC_C,  # C
                ],  # dC/dt
            ]
        )

        der_state_param_t = (der_state_t_state @ y[6:].reshape((6, 5))).flatten()

        der = np.zeros(36)
        der[:6] = der_state_t
        der[6:] = der_state_param_t + der_state_t_param
        return der

    # Call to scipy ODE solver
    ini_solver = np.zeros(36)
    ini_solver[:6] = initial_state
    res = si.solve_ivp(
        am2_ode,
        t_span=(ts[0], ts[-1]),
        y0=ini_solver,
        t_eval=ts[keep_index],
        method=solver_method,
        min_step=min_step,
        **kwargs,
    )

    # Recompute the values of qM, qC
    out = res.y
    der_out = out[6:].reshape((6, 5, out.shape[1]))

    vect0 = np.zeros(out.shape[1])

    X2, S2, C = out[1], out[3], out[5]
    der_X2, der_S2, der_C = der_out[1].T, der_out[3].T, der_out[5].T

    mu2 = mu2max * (S2 / (KS2 + S2 * (1 + S2 / KI2)) - 0.1)

    d_mu2_mu2max = S2 / (KS2 + S2 * (1 + S2 / KI2)) - 0.1
    d_mu2_KS2 = -mu2max * S2 / (KS2 + S2 * (1 + S2 / KI2)) ** 2
    d_mu2_KI2 = mu2max * (S2**3 / (KI2 * (KS2 + S2) + S2**2) ** 2)

    d_mu2_S2 = mu2max * (
        1 / (KS2 + S2 * (1 + S2 / KI2))
        - S2 * (1 + 2 * S2 / KI2) / (KS2 + S2 * (1 + S2 / KI2)) ** 2
    )

    der_mu2 = np.array(
        [vect0, d_mu2_mu2max, vect0, d_mu2_KS2, d_mu2_KI2]
    )  # Change 0 into lists of adequate shape
    qM = k6 * mu2 * X2

    der_qM = (k6 * mu2 * der_X2.T + k6 * (der_mu2 + d_mu2_S2 * der_S2.T) * X2).T

    CO2 = C / (1 + 10 ** (influent_state[keep_index, pH_col] - pKb))
    der_CO2 = (der_C.T / (1 + 10 ** (influent_state[keep_index, pH_col] - pKb))).T

    # CO2 = out[5] + out[3] - out[4]
    # der_CO2 = der_C + der_S2 - der_out[4].T

    phi = CO2 + KH * P_T + qM / kLa
    der_phi = der_CO2 + der_qM / kLa

    KH_PC = (phi - np.sqrt(phi**2 - 4 * KH * P_T * CO2)) / 2

    d_KHPC_phi = (1 - phi / np.sqrt(phi**2 - 4 * KH * P_T * CO2)) / 2
    d_KHPC_CO2 = KH * P_T / (np.sqrt(phi**2 - 4 * KH * P_T * CO2))

    der_KHPC = (d_KHPC_phi * der_phi.T + d_KHPC_CO2 * der_CO2.T).T

    qC = kLa * (CO2 - KH_PC)

    der_qC = kLa * (der_CO2 - der_KHPC)

    derivative = np.zeros((8, 5, out.shape[1]))

    derivative[:6] = out[6:].reshape((6, 5, out.shape[1]))
    derivative[6] = der_qM.T
    derivative[7] = der_qC.T

    return (
        np.array(
            [ts[keep_index], out[0], out[1], out[2], out[3], out[4], out[5], qM, qC]
        ).T,
        np.transpose(derivative, (1, 2, 0)),
    )


def am2_derivative(
    param: DigesterParameter,
    influent_state: DigesterFeed,
    initial_state: DigesterState,
    params_to_der: list = None,
    log_am2=True,
    am2_out: Optional[DigesterStates] = None,  # pylint: disable=W0613
    rel_step: Optional[float] = 10**-7,  # pylint: disable=W0613
    parallel=True,  # pylint: disable=W0613
    **kwargs,
) -> np.ndarray:
    """
    Compute the Jacobian of (log-)AM2 close to param
    for parameters dimension in params_to_der. Time output is dropped.

    The derivative with respect to the parameter is computed by solving an ODE with scipy.
    See documentation of am2_with_der function.

    Some disregarded parameters are included to keep the notations coherent with ADM1 - for which
    a numeric derivation scheme is used.

    Args:
        - param, the parameter at which the derivative of AM2 is to be computed
        - params_to_der, the list of parameter dimensions on which to compute the derivative
        - digester_info, solver_method -> see run_am2 doc
        - log_am2: if False, computes the derivative of AM2, if True, computes the derivative of
            log(AM2). True by default.
        - am2_out: Optional output of AM2 at param (even if log_am2 is True). Disregarded
        - rel_step, parallel: disregarded (kept for similarity with ADM1 module)

    Returns:
        The jacobian in shape P, T, K (P number of parameters, T number of time points, K number of observations)
    """
    # Compute derivative
    preds, der_preds = am2_with_der(
        param, influent_state=influent_state, initial_state=initial_state, **kwargs
    )

    if log_am2:
        der_preds = der_preds / preds[:, 1:]

    if params_to_der is not None:
        params_index = [parameter_dict[name] for name in params_to_der]
        der_preds = der_preds[params_index]

    return der_preds


def am2_derivative_archive(
    param: DigesterParameter,
    influent_state: DigesterFeed,
    initial_state: DigesterState,
    params_to_der: list = None,
    log_am2=True,
    am2_out: Optional[DigesterStates] = None,  # pylint: disable=W0613
    rel_step: Optional[float] = 10**-7,
    max_step: float = 0.5 / (60 * 24),
    parallel=True,
    **kwargs,
) -> np.ndarray:
    """
    Compute the Jacobian of (log-)AM2 close to param
    for parameters dimension in params_to_der. Time output is dropped.
    Built on top of num_der

    To avoid computation issues, relatives increments are considered to the parameter when
    computing the derivative (i.e. we consider AM2(param_j * (1 + epsilon)) - AMD1(param_j)).
    The difference is then corrected for the value of param_j so that the output gives absolute
    derivative.

    With FUN = AM2 or log(AM2) depending on log_am2, computes the derivative by approximating:
    (FUN(param + param_i * epsilon)) - FUN(param))/(epsilon * param_i)

    Args:
        - param, the parameter at which the derivative of AM2 is to be computed
        - params_to_der, the list of parameter dimensions on which to compute the derivative
        - digester_info, solver_method -> see run_am2 doc
        - log_am2: if False, computes the derivative of AM2, if True, computes the derivative of log(AM2).
            True by default.
        - am2_out: Optional output of AM2 at param (even if log_am2 is True). If False, not needed

    Returns:
        The jacobian in shape P, T, K (P number of parameters, T number of time points, K number of observations)

    Note:
        The default max_step argument of run_am2 is set to half a minute. This is necessary to neutralize
        the effect of the max_step argument for small variation on parameter (spurious high frequency, small amplitude
        perturbations which impact the quality of the derivative). For similar reasons, one ought to be careful not to set the
        rel_step too low (or force the max_step to be low as well).
    """
    set_am2 = partial(
        run_am2,
        influent_state=influent_state,
        initial_state=initial_state,
        max_step=max_step,
        **kwargs,
    )

    if log_am2:
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

    # Modify AM2
    loc_am2 = translate_par(post_mod(set_am2))

    print(f"Derivating AM2 around\n{param_to_pd(param)}\nAlong\n{params_to_der}")
    der = num_der(loc_am2, x0=ini_param, rel_step=rel_step, parallel=parallel)

    return der
