"""
PyAM2 - Anaerobic digestion modelisation with Anerobic Model 2 (AM2)

Details about the modelisation can be found at https://doi.org/10.1002/bit.10036 .

The implementation follows the description of the above reference with a few modifications. See
run_AM2 documentation for more details.

PyAM2 is designed for:

    - Modelisation (run_am2)
    - Calibration (submodule optim)
    - Uncertainty Quantification (submodule UQ)

Permanent data storage is organised around submodule IO.
Anaerobic Digestion objects (DigesterInfo, DigesterFeed, DigesterParameter, DigesterState,
DigesterStates) are stored as human readable files (csv, json) and can be loaded/saved using
load_(dig_feed/dig_info/dig_param/dig_state/dig_states) and save methods.

Main functions:
    run_am2 (AM2 modeling of digester from initial state, digester information, digester
        parameter and influent)
    am2_err (measures the difference between predictions and observations)
    score_param (measures the difference between predictions using a specific parameters and
        observations)
    am2_derivative (computes the derivative of AM2 with respect to the digester parameter)
"""

from . import IO, UQ, optim
from .der_am2 import am2_derivative, am2_with_der
from .prediction_error import am2_err, score_param
from .proba import distr_param_indexes, distr_param_map, ref_distr, ref_distr_param
from .run_am2 import run_am2
from .var_buq import am2_var_bayes_uq
