"""
Uncertainty quantification submodule for AM2

Note that the submodule PyAM2.UQ.plot is not loaded by default to avoid dependancy on
unusual packages such as alphashape.
"""
from .beale import am2_beale, am2_beale_pval
from .bootstrap import am2_bootstrap, am2_lin_bootstrap
from .fim import am2_fim, am2_fim_pred, am2_fim_pval
