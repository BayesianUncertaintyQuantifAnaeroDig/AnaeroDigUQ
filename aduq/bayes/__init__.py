""" Bayesian inspired algorithm for joint calibration and uncertainty quantification
Main functions:
- iter_prior, inspired by A. Leurent and R. Moscoviz (https://doi.org/10.1002/bit.28156)
- iter_prior_vi, adaptation of iter_prior to the context of Variational inference for gaussians
- variational_inference, based on Catoni's bound (see https://doi.org/10.48550/arXiv.2110.11216)
"""

from .iter_prior import OptimResultPriorIter, iter_prior, iter_prior_vi
from .variational_inference import (
    AccuSampleVal,
    AccuSampleValDens,
    HistVILog,
    OptimResultVI,
    variational_inference,
)
