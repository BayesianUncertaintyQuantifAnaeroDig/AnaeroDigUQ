"""
Standard distributions and distribution maps for Variational inference and other Bayesian methods.

Distributions are defined on the Free Digester Parameter space (i.e. log space).
The transform maps 0 to the default digester parameter, so that the means of default distributions
 should be 0.

Default standard deviations for each parameter are defined in the _normalisation_param.py file in
the IO submodule.
"""

import numpy as np

from ...proba import FixedCovGaussianMap, GaussianMap, TensorizedGaussianMap
from ..IO import parameter_dict
from ..IO._normalisation_param import std_devs

n_param = len(parameter_dict)

# ------- Standard full gaussian distribution -------

distr_param_map = GaussianMap(n_param)
ref_distr_param = np.zeros((n_param + 1, n_param))
ref_distr_param[1:] = np.diag(std_devs)

ref_distr = distr_param_map(ref_distr_param)

# Use the covariance matrix for optimisation
default_proposal_cov = ref_distr.cov

# ------- Standard diagonal gaussian distribution -------
distr_param_t_map = TensorizedGaussianMap(sample_dim=n_param)
ref_t_distr_param = np.zeros((2, n_param))
ref_t_distr_param[1] = std_devs

ref_t_distr = distr_param_t_map(ref_t_distr_param)

# ------- Standard fixed covariance gaussian distribution -------
distr_param_fcov_map = FixedCovGaussianMap(
    sample_dim=n_param, cov=np.diag(std_devs**2)
)
ref_fcov_param = np.zeros(n_param)

# ------- Group diagonal covariance - Indexes to train -------

parameter_families = [
    ["k_dis", "k_hyd_ch", "k_hyd_pr", "k_hyd_li"],
    ["k_m_su", "K_S_su"],
    ["k_m_aa", "K_S_aa"],
    ["pH_UL:LL_aa", "pH_LL_aa"],
    ["k_m_fa", "K_S_fa", "K_I_h2_fa"],
    ["k_m_c4", "K_S_c4", "K_I_h2_c4"],
    ["k_m_pro", "K_S_pro", "K_I_h2_pro"],
    ["k_m_ac", "K_S_ac"],
    ["pH_UL:LL_ac", "pH_LL_ac", "K_I_nh3"],
    ["k_m_h2", "K_S_h2"],
    ["pH_UL:LL_h2", "pH_LL_h2"],
    ["k_dec"],
    ["K_S_IN"],
]

# Check that no parameter is forgotten
assert np.sum([len(g) for g in parameter_families]) == len(parameter_dict)


def group_to_distr_index(group):
    indexes = [parameter_dict[param] for param in group]
    accu = indexes.copy()
    while len(indexes) > 0:
        k = indexes.pop()
        accu.append(n_param + n_param * k + k)
        for j in indexes:
            accu.append(n_param + n_param * k + j)
            accu.append(n_param + n_param * j + k)
    return accu


distr_param_indexes = []
for g in parameter_families:
    distr_param_indexes += group_to_distr_index(g)
distr_param_indexes.sort()
