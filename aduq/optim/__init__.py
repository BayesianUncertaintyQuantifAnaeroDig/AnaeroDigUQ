"""
Optimisation module

Main functions:
- optim_CMA, a CMA-ES based optimisation algorithm
- optim_MH, optimisation algorithm based on Metropolis-Hastings algorithm

Main classes:
- OptimResult, main class for output of optimisation algorithm
"""

from .optim_cma import CMA_optimiser, optim_CMA
from .optim_mh import optim_MH
from .optim_result import OptimResult, OptimResultCMA
