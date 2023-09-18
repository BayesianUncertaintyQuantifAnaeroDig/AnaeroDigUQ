"""
Helpers used to convert numpy.ndarray objects to more readable pandas object.
Specifies maps between indexes and represented quantities, as well as units.

Note that run_adm1 converts pH information into concentration during the routine.
"""

import warnings
from typing import List, Union

import numpy as np
import pandas as pd


def sort_dict(dico: dict) -> dict:
    """Helper function to insure that the dictionnaries are sorted along their values."""
    return dict(sorted(dico.items(), key=lambda a: a[1]))


## ------------------- Feed data -------------------
influent_state_col = sort_dict(
    {
        "time": 0,  # Day
        "S_su": 1,  # kgCOD M-3
        "S_aa": 2,  # kgCOD M-3
        "S_fa": 3,  # kgCOD M-3
        "S_va": 4,  # kgCOD M-3
        "S_bu": 5,  # kgCOD M-3
        "S_pro": 6,  # kgCOD M-3
        "S_ac": 7,  # kgCOD M-3
        "S_h2": 8,  # kgCOD M-3
        "S_ch4": 9,  # kgCOD M-3
        "S_IC": 10,  # kgCOD M-3
        "S_IN": 11,  # kmole N M-3
        "S_I": 12,  # kmole C M-3
        "X_c": 13,  # kgDCO m-3
        "X_ch": 14,  # kgDCO m-3
        "X_pr": 15,  # kgCOD M-3
        "X_li": 16,  # kgCOD M-3
        "X_su": 17,  # kgCOD M-3
        "X_aa": 18,  # kgCOD M-3
        "X_fa": 19,  # kgCOD M-3
        "X_c4": 20,  # kgCOD M-3
        "X_pro": 21,  # kgCOD M-3
        "X_ac": 22,  # kgCOD M-3
        "X_h2": 23,  # kgCOD M-3
        "X_I": 24,  # kgCOD M-3
        "S_cation": 25,  # kmole M-3
        "S_anion": 26,  # kmole M-3
        "Q": 27,  # M3 Day-1
    }
)

influent_state_units = {
    "time": "Day",
    "S_su": "kgCOD M-3",
    "S_aa": "kgCOD M-3",
    "S_fa": "kgCOD M-3",
    "S_va": "kgCOD M-3",
    "S_bu": "kgCOD M-3",
    "S_pro": "kgCOD M-3",
    "S_ac": "kgCOD M-3",
    "S_h2": "kgCOD M-3",
    "S_ch4": "kgCOD M-3",
    "S_IC": "kmole C M-3",
    "S_IN": "kmole N M-3",
    "S_I": "kgCOD M-3",
    "X_c": "kgCOD M-3",
    "X_ch": "kgCOD M-3",
    "X_pr": "kgCOD M-3",
    "X_li": "kgCOD M-3",
    "X_su": "kgCOD M-3",
    "X_aa": "kgCOD M-3",
    "X_fa": "kgCOD M-3",
    "X_c4": "kgCOD M-3",
    "X_pro": "kgCOD M-3",
    "X_ac": "kgCOD M-3",
    "X_h2": "kgCOD M-3",
    "X_I": "kgCOD M-3",
    "S_cation": "kmole M-3",
    "S_anion": "kmole M-3",
    "Q": "M3 Day-1",
}

## ------------------- Observations/Predictions data -------------------
pred_col = sort_dict(
    {
        "time": 0,
        "S_su": 1,
        "S_aa": 2,
        "S_fa": 3,
        "S_va": 4,
        "S_bu": 5,
        "S_pro": 6,
        "S_ac": 7,
        "S_h2": 8,
        "S_ch4": 9,
        "S_IC": 10,
        "S_IN": 11,
        "S_I": 12,
        "X_c": 13,
        "X_ch": 14,
        "X_pr": 15,
        "X_li": 16,
        "X_su": 17,
        "X_aa": 18,
        "X_fa": 19,
        "X_c4": 20,
        "X_pro": 21,
        "X_ac": 22,
        "X_h2": 23,
        "X_I": 24,
        "S_cation": 25,
        "S_anion": 26,
        "pH": 27,
        "S_va_ion": 28,
        "S_bu_ion": 29,
        "S_pro_ion": 30,
        "S_ac_ion": 31,
        "S_hco3_ion": 32,
        "S_nh3": 33,
        "S_gas_h2": 34,
        "S_gas_ch4": 35,
        "S_gas_co2": 36,
        "S_co2": 37,
        "S_nh4_ion": 38,
        "q_gas": 39,
        "q_ch4": 40,
        "p_ch4": 41,
        "p_co2": 42,
        "VS": 43,
        "VSR": 44,
    }
)

predict_units_dict = {
    "time": "Day",
    "S_su": "kgCOD M-3",
    "S_aa": "kgCOD M-3",
    "S_fa": "kgCOD M-3",
    "S_va": "kgCOD M-3",
    "S_bu": "kgCOD M-3",
    "S_pro": "kgCOD M-3",
    "S_ac": "kgCOD M-3",
    "S_h2": "kgCOD M-3",
    "S_ch4": "kgCOD M-3",
    "S_IC": "kmole C M-3",
    "S_IN": "kmole N M-3",
    "S_I": "kgCOD M-3",
    "X_c": "kgCOD M-3",
    "X_ch": "kgCOD M-3",
    "X_pr": "kgCOD M-3",
    "X_li": "kgCOD M-3",
    "X_su": "kgCOD M-3",
    "X_aa": "kgCOD M-3",
    "X_fa": "kgCOD M-3",
    "X_c4": "kgCOD M-3",
    "X_pro": "kgCOD M-3",
    "X_ac": "kgCOD M-3",
    "X_h2": "kgCOD M-3",
    "X_I": "kgCOD M-3",
    "S_cation": "kmole M-3",
    "S_anion": "kmole M-3",
    "pH": "pH",
    "S_va_ion": "kgCOD M-3",
    "S_bu_ion": "kgCOD M-3",
    "S_pro_ion": "kgCOD M-3",
    "S_ac_ion": "kgCOD M-3",
    "S_hco3_ion": "kmole M-3",
    "S_nh3": "kmole M-3",
    "S_gas_h2": "kgCOD M-3",
    "S_gas_ch4": "kgCOD M-3",
    "S_gas_co2": "kmole M-3",
    "S_co2": "kmole M-3",
    "S_nh4_ion": "kmole M-3",
    "q_gas": "M3 Day-1",
    "q_ch4": "M3 Day-1",
    "p_ch4": "bar",
    "p_co2": "bar",
    "VS": "kgVS M-3",
    "VSR": "ratio",
}

# Predictions which are used during fitting
small_predictions_col = [
    "time",
    "S_va",
    "S_bu",
    "S_pro",
    "S_ac",
    "S_IN",
    "q_gas",
    "q_ch4",
    "p_ch4",
    "p_co2",
]

small_predictions_numb = [
    pred_col[predict_type] for predict_type in small_predictions_col
]

small_predictions_rm_numb = list(
    set(pred_col.values()).difference(set(small_predictions_numb))
)  # Columns indexes of predictions not used during fitting

## ------------------- Initial state data -------------------
# To simplify implementation, all predicted features are included in initial state
# but S_co2, S_nh4_ion, q_gas, q_ch4, p_ch4, p_co2, VS and VSR are not used
# and can be set to NaN.
initial_state_col = pred_col

## ------------------- Digester Calibration parameter -------------------
parameter_dict = sort_dict(
    {
        "k_dis": 0,  # Day-1
        "k_hyd_ch": 1,  # Day-1
        "k_hyd_pr": 2,  # Day-1
        "k_hyd_li": 3,  # Day-1
        "k_m_su": 4,  # Day-1
        "k_m_aa": 5,  # Day-1
        "k_m_fa": 6,  # Day-1
        "k_m_c4": 7,  # Day-1
        "k_m_pro": 8,  # Day-1
        "k_m_ac": 9,  # Day-1  ## Typo in original PyADM1, noted as kgCOD M-3
        "k_m_h2": 10,  # Day-1
        "k_dec": 11,  # Day-1
        "K_S_IN": 12,  # M
        "K_S_su": 13,  # kgCOD M-3
        "K_S_aa": 14,  # kgCOD M-3
        "K_S_fa": 15,  # kgCOD M-3
        "K_S_c4": 16,  # kgCOD M-3
        "K_S_pro": 17,  # kgCOD M-3
        "K_S_ac": 18,  # kgCOD M-3
        "K_S_h2": 19,  # kgCOD M-3
        "K_I_h2_fa": 20,  # kgCOD M-3
        "K_I_h2_c4": 21,  # kgCOD M-3
        "K_I_h2_pro": 22,  # kgCOD M-3
        "K_I_nh3": 23,  # M
        "pH_UL:LL_aa": 24,  # pH unit
        "pH_LL_aa": 25,  # pH unit
        "pH_UL:LL_ac": 26,  # pH unit
        "pH_LL_ac": 27,  # pH unit
        "pH_UL:LL_h2": 28,  # pH unit
        "pH_LL_h2": 29,  # pH unit
    }
)

parameter_units = {
    "k_dis": "Day-1",
    "k_hyd_ch": "Day-1",
    "k_hyd_pr": "Day-1",
    "k_hyd_li": "Day-1",
    "k_m_su": "Day-1",
    "k_m_aa": "Day-1",
    "k_m_fa": "Day-1",
    "k_m_c4": "Day-1",
    "k_m_pro": "Day-1",
    "k_m_ac": "Day-1",
    "k_m_h2": "Day-1",
    "k_dec": "Day-1",
    "K_S_IN": "M",
    "K_S_su": "kgCOD M-3",
    "K_S_aa": "kgCOD M-3",
    "K_S_fa": "kgCOD M-3",
    "K_S_c4": "kgCOD M-3",
    "K_S_pro": "kgCOD M-3",
    "K_S_ac": "kgCOD M-3",
    "K_S_h2": "kgCOD M-3",
    "K_I_h2_fa": "kgCOD M-3",
    "K_I_h2_c4": "kgCOD M-3",
    "K_I_h2_pro": "kgCOD M-3",
    "K_I_nh3": "M",
    "pH_UL:LL_aa": "pH",
    "pH_LL_aa": "pH",
    "pH_UL:LL_ac": "pH",
    "pH_LL_ac": "pH",
    "pH_UL:LL_h2": "pH",
    "pH_LL_h2": "pH",
}

n_params = len(parameter_dict)


## ------------------- VS/COD conversions -------------------
COD_VS = {
    "S_su": 1.03,
    "S_aa": 1.5,
    "S_fa": 2.0,
    "X_c": 1.42,
    "X_ch": 1.03,
    "X_pr": 1.5,
    "X_li": 2.0,
    "X_su": 1.42,
    "X_aa": 1.42,
    "X_fa": 1.42,
    "X_c4": 1.42,
    "X_pro": 1.42,
    "X_ac": 1.42,
    "X_h2": 1.42,
    "X_I": 1.5,
}  # gCOD / gVS

cod_vs_dig_states_cols = [pred_col[x] for x in COD_VS.keys()]

cod_vs_feed_cols = [influent_state_col[x] for x in COD_VS.keys()]

cod_vs_values = np.array(list(COD_VS.values()))


## ------------------- Helper functions for parameters -------------------
def param_names_to_index(param_names: List[str]) -> List[int]:
    """Transform a list of parameter names into the indexes of those parameters"""
    return [parameter_dict[param_name] for param_name in param_names]


def pred_names_to_index(pred_col_names: list[str]) -> list[int]:
    """Transform a list of prediction names into the indexes of those predictions"""
    return [pred_col[name] for name in pred_col_names]


def param_to_numpy(param: Union[dict, pd.Series]) -> np.ndarray:
    """Transform a parameter coded as a dictionnary or pandas.Series into a np.ndarray"""
    if isinstance(param, dict):
        return pd.Series(param[parameter_dict.keys()]).to_numpy()
    # Assumes param is a pd.Series
    return param[parameter_dict.keys()].to_numpy()


def param_to_pandas(param: np.ndarray) -> pd.Series:
    """Transform a parameter coded as a np.ndarray (or DigesterParameter) into a panda Series"""
    return pd.Series(param, index=parameter_dict.keys())


## ------------------- Helper functions for feed -------------------
def influent_to_numpy(influent_state: pd.DataFrame) -> np.ndarray:
    """
    Transforms feed data stored as a pandas DataFrame into a suitable numpy.ndarray (columns
    are reordered to match the adequate order if necessary)
    """
    if not set(influent_state.columns).issubset(set(influent_state_col.keys())):
        missing = set(influent_state_col.keys()).difference(influent_state.columns)
        raise Exception(
            f"influent_state should have the following columns: {list(missing)}"
        )
    return influent_state[influent_state_col.keys()].to_numpy()


def influent_to_pandas(influent_state: np.ndarray) -> pd.DataFrame:
    """
    Transforms a correctly formated feed data stored as a numpy.ndarray
    or DigesterFeed into a user-friendly pandas DataFrame object.
    """
    return pd.DataFrame(influent_state, columns=influent_state_col.keys())


## ------------------- Helper functions for observations/predictions -------------------
def states_to_numpy(dig_states: pd.DataFrame) -> np.ndarray:
    """
    Convert Observations/predictions data as pandas DataFrame into np.ndarray.

    If columns are missing, NaNs are filled. Throws a warning if columns in
    small_predictions_col are missing.
    """
    # Check if any major type of predictions is missing and raise a warning if true
    missing_major = set(small_predictions_col).difference(set(dig_states.columns))
    if len(missing_major) > 0:
        warnings.warn(
            f"Some important types of observation are missing:\n{list(missing_major)}"
        )

    # Complete dataframe for missing type of observations
    missing_cols = set(pred_col.keys()).difference(set(dig_states.columns))
    for missing_col in missing_cols:
        dig_states[missing_col] = np.NaN

    cols = list(pred_col.keys())
    return dig_states[cols].to_numpy()


def states_to_pandas(dig_states: np.ndarray) -> pd.DataFrame:
    """
    Observations/predictions data as np.ndarray into pandas DataFrame.

    Raises Exceptions if input cannot be interpreted correctly.
    """

    if len(dig_states.shape) != 2:
        raise Exception("dig_states must be 2 dimensional")

    if dig_states.shape[1] != len(pred_col):
        raise Exception("Could not infer prediction format")

    return pd.DataFrame(dig_states, columns=pred_col.keys())


def init_state_to_numpy(init_state: pd.Series) -> np.ndarray:
    """
    Reorder initial_state columns if needed and transform it to 1D np.ndarray.
    """
    keys_name = list(pred_col.keys())  # in proper order
    missing = set(keys_name).difference(set(init_state.keys()))
    if len(missing) > 0:
        warnings.warn(f"Keys {missing} are missing from init_state. Set to NaN")
        for name in missing:
            init_state[name] = np.NaN
    return init_state[keys_name].to_numpy().flatten()

    # return init_state[initial_state_col.keys()].to_numpy().flatten()
