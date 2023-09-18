from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _plot_2D_sample(
    sample, i, j, inds=None, weights=None, plot_obj=plt, c="tab:orange", **kwargs
):
    if i != j:
        if inds is not None:
            sample = sample[inds]
        plot_obj.scatter(sample[:, i], sample[:, j], c=c, **kwargs)
    else:
        sns.kdeplot(x=sample[:, i], weights=weights, ax=plot_obj, color=c)


def grid_plot_2D_sample(
    fig, ax, sample: np.ndarray, weights: Optional[List] = None, marker=".", **kwargs
):
    if weights is None:
        inds = None
    else:
        inds = np.random.choice(range(len(weights)), 50000, p=weights)
    sample = np.array(sample)

    n = sample.shape[1]

    for i in range(n):
        for j in range(n):

            _plot_2D_sample(
                sample,
                i=i,
                j=j,
                inds=inds,
                weights=weights,
                plot_obj=ax[j, i],
                marker=marker,
                **kwargs
            )

    fig.tight_layout()

    return fig, ax
