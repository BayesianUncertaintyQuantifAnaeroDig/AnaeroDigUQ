"""
Modelisation of Anaerobic digestion with AM2

Implementation derived from original AM2 description by Bernard et al. 2001
(https://doi.org/10.1002/bit.10036).

Following Hassam et al. 2015 (https://doi.org/10.1016/j.bej.2015.03.007), a mortality rate of 0.1
was added when computing growth rate. 

The equation to compute CO2 was also modified to account for the knowledge of the pH:

CO2 = C / (
    1 + 10 ** (pH - pKb)
)

which amounts to eq 53 in Bernard et al. 2001 or equivalently from combining eq 3. and 5. from
the same source.



Main Input:
    param, description of the microbiology which is to be calibrated
    influent_state, description of what is fed the digester
    initial_state, description of what is inside the digester at the beginning
Further argument:
    solver_method, the solver by scipy to solve the ODE. Default is LSODA
    min_step, the minimal time increment for the solver (to avoid long simulation time)
    max_step, the maximum time increment for the solver (to force good precision)
"""


import warnings

import numpy as np
import scipy.integrate as si

from ._typing import DigesterFeed, DigesterParameter, DigesterState, DigesterStates
from .IO import influent_state_col

# ------------ Prepare constants ------------

# Hassam 2015
# k1 = 23.0 # gCOD gVS^{-1}
# k2 = 464.0 # mmol gVS^{-1}
# k3 = 514.0 # mmol gVS^{-1}
# k4 = 310.0 # mmol gVS^{-1}
# k5 = 600.0 # mmol gVS^{-1}
# k6 = 253.0 # mmol gVS^{-1}

# kLa = 24.0 # day-1

# Bernard 2001
k1 = 42.14  # gCOD gVS^{-1}
k2 = 116.5  # mmol gVS^{-1}
k3 = 268.0  # mmol gVS^{-1}
k4 = 50.6  # mmol gVS^{-1}
k5 = 343.6  # mmol gVS^{-1}
k6 = 453.0  # mmol gVS^{-1}

kLa = 19.8  # day-1

KH = 26.7  # mmol L^{-1} atm^{-1}
pKb = -np.log10(6.5 * 10 ** (-7))  # Kb in mol/L

P_T = 1.0  # bar \simeq 1 atm

alpha = 1.0

# ------------ helper functions ------------


def day_index(ts: np.ndarray):
    """Given time stamps ts (in float), returns index where a new day is started"""
    u = np.zeros(len(ts), dtype=int)
    loc_day = int(ts[0]) + 1
    compt = 0
    for i, t in enumerate(ts):
        if t >= loc_day:
            u[compt] = i
            loc_day += 1
            compt += 1
    return u[:compt]


pH_col = influent_state_col["pH"]


class ParamHandling(Warning):
    pass


# ------------ Main function ------------
def run_am2(
    param: DigesterParameter,
    influent_state: DigesterFeed,
    initial_state: DigesterState,
    solver_method: str = "LSODA",
    min_step: float = 10**-4,
    **kwargs
) -> DigesterStates:
    """
    Models digester evolution using AM2.

    Default solver for differential equation is LSODA.
    The step size is inferred from the feed file, with a maximum value of 15 minute, and a minum value of 20 seconds

    Output is a np.ndarray.
        First column is time (one information per day),
        Remaining columns are
            "X1", in gVS L-1 # conc. acidogenic bacteria
            "X2", in gVS L-1 # conc. methanogenic bacteria
            "S1", in gCOD L-1 # conc. substrate
            "S2", in mmol L-1 # conc. VFA
            "Z", in mmol L-1 # tot. alkalinity
            "C", in mmol L-1 # tot. inorg carbon conc.
            "qm", in mmol L-1 Day-1 # methane flow
            "qc", in mmol L-1 Day-1 # carbon dioxide flow
    """

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

        dS/dt (t) = am2_ode(t, S)
        """

        # Read feed information
        index = min(np.sum(t > ts), influent_state.shape[0] - 1)
        D, S1_in, S2_in, Z_in, C_in, pH = influent_state[index, 1:]

        # Unpack current digester state information
        X1, X2, S1, S2, Z, C = y

        # Compute intermediary
        mu1 = mu1max * (S1 / (KS1 + S1) - 0.1)  # D-1  # Hassam Eq 6
        mu2 = mu2max * (S2 / (KS2 + S2 * (1 + S2 / KI2)) - 0.1)  # D-1 # Hassam Eq 7

        qM = k6 * mu2 * X2  # mmol L-1 D-1 # Bernard Eq 28
        CO2 = C / (
            1 + 10 ** (pH - pKb)
        )  # Alternative to Bernard used since pH is known

        phi = (
            CO2 + KH * P_T + qM / kLa
        )  # mmol L-1 # Below Bernard Eq 27 + Bernard Eq 19

        KH_PC = (
            phi - np.sqrt(phi**2 - 4 * KH * P_T * CO2)
        ) / 2  # mmol L-1 # from Eq 27

        qC = kLa * (CO2 - KH_PC)  # mmol L-1 D-1 # from Bernard Eq 26 + Bernard Eq 19

        return [
            (mu1 - alpha * D) * X1,  # dX1 # g L-1 D-1 # Bernard Eq 20
            (mu2 - alpha * D) * X2,  # dX2 # mmol L-1 D-1 # Bernard Eq 21
            D * (S1_in - S1) - k1 * mu1 * X1,  # dS1 # gCod L-1 D-1 # Bernard Eq 23
            D * (S2_in - S2)
            + k2 * mu1 * X1
            - k3 * mu2 * X2,  # dS2 # mmol L-1 D-1 # Bernard Eq 24
            D * (Z_in - Z),  # dZ # mmol L-1 D-1 # Bernard Eq 22
            D * (C_in - C)
            - qC
            + k4 * mu1 * X1
            + k5 * mu2 * X2,  # dC # mmol L-1 D-1 # Bernard Eq 25
        ]

    # Call to scipy ODE solver
    res = si.solve_ivp(
        am2_ode,
        t_span=(ts[0], ts[-1]),
        y0=np.array(initial_state),
        t_eval=ts[keep_index],
        method=solver_method,
        min_step=min_step,
        **kwargs
    )

    # Recompute the values of qM, qC
    out = res.y

    mu2 = mu2max * (out[3] / (KS2 + out[3] * (1 + out[3] / KI2)) - 0.1)
    qM = k6 * mu2 * out[1]

    CO2 = out[5] / (1 + 10 ** (influent_state[keep_index, pH_col] - pKb))
    # CO2 = out[5] + out[3] - out[4]
    phi = CO2 + KH * P_T + qM / kLa

    KH_PC = (phi - np.sqrt(phi**2 - 4 * KH * P_T * CO2)) / 2

    qC = kLa * (CO2 - KH_PC)

    return np.array(
        [ts[keep_index], out[0], out[1], out[2], out[3], out[4], out[5], qM, qC]
    ).T
