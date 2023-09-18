"""
Default parameter values and default uncertainty on values
"""
import numpy as np
import pandas as pd

from ._helper_pd_np import parameter_dict

# Taken from Rosen & Jeppsson
renorm_param = {
    "k_dis": 0.5,
    "k_hyd_ch": 10,
    "k_hyd_pr": 10,
    "k_hyd_li": 10,
    "k_m_su": 30,
    "k_m_aa": 50,
    "k_m_fa": 6,
    "k_m_c4": 20,
    "k_m_pro": 13,
    "k_m_ac": 8,
    "k_m_h2": 35,
    "k_dec": 0.02,
    "K_S_IN": 0.0001,
    "K_S_su": 0.5,
    "K_S_aa": 0.3,
    "K_S_fa": 0.4,
    "K_S_c4": 0.2,
    "K_S_pro": 0.1,
    "K_S_ac": 0.15,
    "K_S_h2": 7e-6,
    "K_I_h2_fa": 5e-6,
    "K_I_h2_c4": 1e-5,
    "K_I_h2_pro": 3.5e-6,
    "K_I_nh3": 0.0018,
    "pH_UL:LL_aa": 1.5,
    "pH_LL_aa": 4.0,
    "pH_UL:LL_ac": 1.0,
    "pH_LL_ac": 5.0,
    "pH_UL:LL_h2": 1.0,
    "pH_LL_h2": 5.0,
}


max_param = {key: 10 * renorm_param[key] for key in renorm_param.keys()}
for name in ["pH_LL_aa", "pH_LL_ac", "pH_LL_h2"]:
    max_param[name] = 6.0
for name in ["pH_UL:LL_aa", "pH_UL:LL_ac", "pH_UL:LL_h2"]:
    max_param[name] = 3.0

# Set up uncertainty from initial ADM1 report
# The uncertainty is reported as a %, and as such in log space.
devs_dict = {
    "k_dis": 3,
    "k_hyd_ch": 2,
    "k_hyd_pr": 2,
    "k_hyd_li": 3,
    "k_m_su": 2,
    "k_m_aa": 2,
    "k_m_fa": 3,
    "k_m_c4": 2,
    "k_m_pro": 2,
    "k_m_ac": 2,
    "k_m_h2": 2,
    "k_dec": 2,
    "K_S_IN": 1,
    "K_S_su": 2,
    "K_S_aa": 1,
    "K_S_fa": 3,
    "K_S_c4": 3,
    "K_S_pro": 2,
    "K_S_ac": 2,
    "K_S_h2": 2,
    "K_I_h2_fa": 1,
    "K_I_h2_c4": 1,
    "K_I_h2_pro": 1,
    "K_I_nh3": 1,
    "pH_UL:LL_aa": 2,
    "pH_LL_aa": 2,
    "pH_UL:LL_ac": 1,
    "pH_LL_ac": 1,
    "pH_UL:LL_h2": 2,
    "pH_LL_h2": 1,
}

devs_interp = {1: 0.3, 2: 1.0, 3: 3.0}

# The report indicate range, while we consider standard deviation. To correct for this,
# we use a corrective factor of 0.4.

std_devs = 0.4 * np.array(
    [np.log(devs_interp[devs_dict[name]] + 1) for name in parameter_dict]
)

# We also define ranges for global sensibility analysis.
min_col = -2 * std_devs
max_col = 2 * std_devs

param_range = pd.DataFrame(
    {"min": min_col, "max": max_col}, index=list(parameter_dict.keys())
)
