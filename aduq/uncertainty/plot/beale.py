from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def beale_plot(beale_bound: Union[np.ndarray, pd.DataFrame]):

    # Infer names of parameters
    if isinstance(beale_bound, pd.DataFrame):
        names = beale_bound.columns
        col_names = beale_bound.columns
    else:
        names = [f"Column {i + 1}" for i in range(beale_bound.shape[1])]
        col_names = list(range(beale_bound.shape[1]))

    # Prepare subplots
    fig, ax = plt.subplots(len(names), len(names))

    # Fill subplots
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ax[i, j].plot(beale_bound[col_names[i]], beale_bound[col_names[j]], ",")
    for axe, name in zip(ax[0], names):
        axe.set_title(name)

    # labels
    for axe, name in zip(ax[:, 0], names):
        axe.set_ylabel(name)

    fig.tight_layout()
    return fig, ax
