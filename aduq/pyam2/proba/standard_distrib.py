"""
Note:
    0.4 factor for covariance can be explained in the following way:

        Probability of X > k * m = Probability of (X - m) / m > (k-1) 
        If X is normally distributed N(m, m * sigma), then this is
        Proba N(0,1) > (k-1)/ sigma
        If X is log normal distributed, N(log(m), a), then this it
        Proba N(0,1) > log(k) /a
        Hence for the probabilities to match, a = sigma log(k)/(k-1)

        Take k = 3 to conclude.
"""

import numpy as np

from ...proba import FixedCovGaussianMap, GaussianMap, TensorizedGaussianMap
from ..IO import parameter_dict

n_param = len(parameter_dict)


distr_param_map = GaussianMap(sample_dim=n_param)


ref_distr_param = np.zeros((n_param + 1, n_param))



std_devs = 0.4 * np.array([0.5, 1.2, 0.8, 1.2, 1.0])
ref_distr_param[1:] = np.diag(std_devs)

ref_distr = distr_param_map(ref_distr_param)

distr_param_t_map = TensorizedGaussianMap(sample_dim=n_param)
ref_t_distr_param = np.zeros((2, n_param))

ref_t_distr_param[1] = std_devs

ref_t_distr = distr_param_t_map(ref_t_distr_param)

distr_param_fcov_map = FixedCovGaussianMap(
    sample_dim=n_param, cov=np.diag(std_devs**2)
)
ref_fcov_param = np.zeros(n_param)

parameter_families = [
    ["mu1max", "KS1"],
    ["mu2max", "KS2", "KI2"],
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
