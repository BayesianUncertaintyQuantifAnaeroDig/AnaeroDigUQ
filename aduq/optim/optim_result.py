""" Optimisation result classes """

from typing import List, Optional

import numpy as np


class OptimResult:
    """
    Main class for output of optimization routines

    This class functions as an organized storage of optimisation related variables. These include
    - opti_param, the parameter returned by the optimisation routine
    - converged, whether the optimisation routine assumes convergence
    - opti_score, the score achieved by the optimisation routine (Optional)
    - hist_param, the list of parameters in the optimisation route (Optional)
    - hist_score, the scores of the parameters in hist_param (Optional)

    Future:
        Add optional field for a dict containing the calibration algorithm hyperparameters
    """

    def __init__(
        self,
        opti_param,
        converged: bool,
        opti_score: Optional[float] = None,
        hist_param: Optional[list] = None,
        hist_score: Optional[List[float]] = None,
    ):

        self.opti_param = opti_param
        self.converged = converged
        self.opti_score = opti_score
        self.hist_param = hist_param
        self.hist_score = hist_score

    def convert(self, fun):
        """
        Conversion of parameters in OptimResult object

        If J o fun was optimized in order to optimize J, then converts the optimisation result for
        the optimisation of J (i.e. parameters are converted)
        """
        self.opti_param = fun(self.opti_param)
        if self.hist_param is not None:
            self.hist_param = [fun(par) for par in self.hist_param]

    def get_best_param(self):
        if (self.hist_param is not None) and (self.hist_score is not None):
            return self.hist_param[np.argmin(self.hist_score)]
        raise ValueError("Empty hist_param or hist_score attributes")


class OptimResultCMA(OptimResult):
    """
    Subclass of OptimResult for output of CMA-ES optimisaiton algorithm

    This class functions as an organized storage of optimisation related variables. These include
    - opti_param, the parameter returned by the optimisation routine
    - converged, whether the optimisation routine assumes convergence
    - opti_score, the score achieved by the optimisation routine (Optional)
    - hist_param, the list of parameters in the optimisation route (Optional)
    - hist_score, the scores of the parameters in hist_param (Optional)
    """

    def __init__(
        self,
        opti_param,
        converged: bool,
        opti_score: Optional[float] = None,
        hist_param: Optional[list] = None,
        hist_score: Optional[List[float]] = None,
        hist_cov: Optional[list] = None,
    ):
        super().__init__(
            opti_param=opti_param,
            converged=converged,
            opti_score=opti_score,
            hist_param=hist_param,
            hist_score=hist_score,
        )
        self.hist_cov = hist_cov
