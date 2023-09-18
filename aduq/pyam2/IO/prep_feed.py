import numpy as np
import pandas as pd

from .._typing import DigesterFeed
from ._helper import influent_state_col


def convert_feed(feed: pd.DataFrame, pH_feed, V_liq, pH) -> DigesterFeed:
    """
    Convert feed information designed for ADM1 model to feed information for AM2 model

    Input feed should be a dataframe containing the following columns (with units)
        "time": # Day
        "S_su"  # kgCOD M-3
        "S_aa"  # kgCOD M-3
        "S_fa"  # kgCOD M-3
        "S_va" # kgCOD M-3
        "S_bu" # kgCOD M-3
        "S_pro"  # kgCOD M-3
        "S_ac"  # kgCOD M-3
        "S_h2" # kgCOD M-3
        "S_ch4" # kgCOD M-3
        "S_IC" # kgCOD M-3
        "S_IN" # kmole N M-3
        "S_I" # kmole C M-3
        "X_c" # kgDCO m-3
        "X_ch" # kgDCO m-3
        "X_pr" # kgCOD M-3
        "X_li" # kgCOD M-3
        "X_su" # kgCOD M-3
        "X_aa" # kgCOD M-3
        "X_fa" # kgCOD M-3
        "X_c4"  # kgCOD M-3
        "X_pro" # kgCOD M-3
        "X_ac" # kgCOD M-3
        "X_h2" # kgCOD M-3
        "X_I"  # kgCOD M-3
        "S_cation"  # kmole M-3
        "S_anion"  # kmole M-3
        "Q"  # M3 Day-1
    Other arguments
        pH_feed: Measure of pH of entering feed, either a time series of same length as feed,
            or a float
        V_liq: Volume of liquid phase in the digester
        pH: Measure of the pH in the digester, either a time series of same length as feed, or a
            float

    Note:
    Columns "S_h2", "S_ch4", "S_IN", "S_I", "X_I", "S_cation", "S_anion", "Q" are not used and as
        such not necessary. Mentionned for coherence with ADM1 standards

    Output:
        A numpy.ndarray which can be used as DigesterFeed for AM2
    """
    # Prepare ShCO3
    ShCO3 = np.exp(-np.log(10.0) * (14 - pH_feed))

    # Compute AM2 feed columns

    X1_in = (feed["X_su"] + feed["X_aa"] + feed["X_fa"]) / 1.55
    X2_in = (feed["X_ac"] + feed["X_h2"] + feed["X_c4"] + feed["X_pro"]) / 1.55

    S1_in = (
        feed["S_su"]
        + feed["S_aa"]
        + feed["S_fa"]
        + feed["X_c"]
        + feed["X_ch"]
        + feed["X_pr"]
        + feed["X_li"]
    )
    S2_in = (
        (feed["S_va"] / 208)
        + (feed["S_bu"] / 160)
        + (feed["S_pro"] / 112)
        + (feed["S_ac"] / 64)
    ) * 1000

    Z_in = (
        (feed["S_va"] / 208)
        + (feed["S_bu"] / 160)
        + (feed["S_pro"] / 112)
        + (feed["S_ac"] / 64)
        + ShCO3
    ) * 1000
    C_in = feed["S_IC"].to_numpy() * 1000

    D = feed["Q"] / V_liq

    # Prepare dataFrame
    Data = np.array(
        [feed["time"].to_numpy(), D, X1_in, X2_in, S1_in, S2_in, Z_in, C_in]
    ).T
    n_feed = pd.DataFrame(Data, columns=["time", "D", "X1", "X2", "S1", "S2", "Z", "C"])
    n_feed["pH"] = pH

    # Reorder and output as numpy.ndarray
    return n_feed[list(influent_state_col.keys())].to_numpy()
