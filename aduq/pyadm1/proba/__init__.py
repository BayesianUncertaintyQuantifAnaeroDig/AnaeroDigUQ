"""
Distribution maps used in ADM1 related routines.

Main maps are
    distr_param_map (maps to all Gaussian distributions),
    distr_param_t_map (maps to Gaussian distributions with diagonal covariance)
    distr_param_fcov_map (maps to Gaussian distributions with covariance fixed to default)
"""

from .standard_distrib import (
    default_proposal_cov,
    distr_param_fcov_map,
    distr_param_indexes,
    distr_param_map,
    distr_param_t_map,
    ref_distr,
    ref_distr_param,
    ref_fcov_param,
    ref_t_distr,
    ref_t_distr_param,
)
