"""
PyADM1 - Anaerobic digestion modelisation with Anerobic Digestion Model 1 (ADM1)

Details about the modelisation can be found at https://doi.org/10.2166/wst.2002.0292 .
This package is based around ADM1 implementation https://github.com/CaptainFerMag/PyADM1 .

PyADM1 is designed for:

    - Modelisation (run_adm1)
    - Sensitivity Analysis (submodule SA)
    - Calibration (submodule optim)
    - Uncertainty Quantification (submodule UQ)

Permanent data storage is organised around submodule IO.
Anaerobic Digestion objects (DigesterInfo, DigesterFeed, DigesterParameter, DigesterState,
DigesterStates) are stored as human readable files (csv, json) and can be loaded/saved using
load_(dig_feed/dig_info/dig_param/dig_state/dig_states) and save methods.

Main functions:
    run_adm1 (ADM1 modeling of digester from initial state, digester information, digester
        parameter and influent)
    adm1_err (measures the difference between predictions and observations)
    score_param (measures the difference between predictions using a specific parameters and
        observations)
    adm1_derivative (computes the derivative of ADM1 with respect to the digester parameter)
"""

from . import IO, SA, UQ, optim
from .der_adm1 import adm1_derivative
from .prediction_error import adm1_err, score_param
from .proba import distr_param_indexes, distr_param_map, ref_distr_param
from .run_adm1 import run_adm1
from .var_buq import adm1_var_bayes_uq
