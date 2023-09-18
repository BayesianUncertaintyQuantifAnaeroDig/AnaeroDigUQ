"""
VS computations for both intrant and digester.

Added to original ADM1 implementation by (https://github.com/CaptainFerMag/PyADM1) to compute VSR
"""

from typing import Optional

import numpy as np

from .IO import DigesterFeed, DigesterStates
from .IO._helper_pd_np import (
    cod_vs_dig_states_cols,
    cod_vs_feed_cols,
    cod_vs_values,
    influent_state_col,
)

q_col = influent_state_col["Q"]


def feed_vs(
    dig_feed: DigesterFeed, per_day: bool = True, day_begin: Optional[int] = None
) -> np.ndarray:
    """
    Compute volatile solid concentration in the feed.

    Args:
        dig_feed: a DigesterFeed object, on which the VS part is computed
        per_day: should the VS be characterized for a day or between t and dt?
        day_begin: which is the first day in the dig_feed file? (See below).
    If per_day is True:
        As usual, the description of the substrate at line i is assumed to be valid
        from t_{i-1} to t_{i}. No information is accessible therefore for the substrate
        before t_0. For the first and last day, VS is computed using the mean information
        for that day.

    Output in kg VS M-3
    """

    # Get vs_in at each time step
    vs_in = dig_feed[:, cod_vs_feed_cols] @ (1 / cod_vs_values)  # kg VS M-3

    if per_day:
        # Take the mean of the VS in the day.
        q_in = dig_feed[:, q_col]  # M3 Day-1
        t = dig_feed[:, 0]

        day_end = int(t[-1])

        if day_begin is None:
            day_begin = int(t[0]) + 1

        n_days = day_end - day_begin + 1
        vs_per_day = np.zeros(n_days)

        ti = t[0]
        ti1 = t[1]
        vs_accu = 0
        loc_day = 0
        v_accu = 0
        for vs_conc, q, ti1 in zip(vs_in, q_in, t[1:]):
            if ti1 >= (loc_day + day_begin):
                # New day: compute mean value, store result, prepare new value

                # Contribution of vs to loc_day
                vs_accu += (loc_day + day_begin - ti) * q * vs_conc  # kg VS
                v_accu += (loc_day + day_begin - ti) * q  # m3

                # Store value
                vs_per_day[loc_day] = vs_accu / v_accu

                # Reset and Contribution of vs to loc_day + 1
                vs_accu = (ti1 - day_begin + loc_day) * q * vs_conc  # kg VS
                v_accu = (ti1 - day_begin + loc_day) * q  # m3
                loc_day += 1

            else:
                # Contribution of vs to loc_day
                vs_accu += (ti1 - ti) * q * vs_conc  # kg VS
                v_accu += (ti1 - ti) * q  # m3
            ti = ti1  # update value of ti

        # End loop

        # Correct for first day (potentially, t[0] is not an integer
        # so renormalisation is wrong)
        vs_per_day[0] = vs_per_day[0] / (1 + int(t[0]) - t[0])

        return vs_per_day

    else:
        return vs_in


def dig_states_vs(dig_states: DigesterStates) -> np.ndarray:
    """
    Volatile solid in the digester (and by extension in the digester output).
    In kg VS M-3

    """
    return dig_states[:, cod_vs_dig_states_cols] @ (1 / cod_vs_values)  # kgVS M-3
