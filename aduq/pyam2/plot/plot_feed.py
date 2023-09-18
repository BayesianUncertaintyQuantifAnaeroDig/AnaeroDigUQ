import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..IO._helper import influent_state_col, influent_state_units


def plot_feed(*feeds: Tuple[np.ndarray], fold_path: str, aml_run=None) -> None:
    """Saves plots of feeds information inside fold_path
    (names are followed by "_in").

    For AML experiments purpose, an aml_run can also be passed so that the
    plot is also logged.
    """
    influent_state_names = list(influent_state_col.keys())
    for i in range(len(influent_state_col) - 1):
        feed_name = influent_state_names[i + 1]

        for feed in feeds:
            plt.plot(feed[:, 0], feed[:, i + 1])

        plt.xlabel(f"Time ({influent_state_units[0]})")
        if influent_state_units[i + 1] != "":
            unit_text = f" ({influent_state_units[i+1]})"
        else:
            unit_text = ""
        plt.ylabel(f"{feed_name}" + unit_text)

        if aml_run is not None:
            aml_run.log_image(f"{feed_name}_in", plot=plt)

        plt.savefig(os.path.join(fold_path, f"{feed_name}_in.png"))
        plt.clf()
