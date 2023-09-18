import warnings
from typing import List, Optional, Union

import alphashape
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shapely
from sklearn.ensemble import IsolationForest


def get_cov(data: np.ndarray, weights: np.ndarray):
    """Returns the covariance of data.

    Args:
        data, a np.ndarray of shape (N, d1, ..., dn)
        weights, a np.ndarray of shape (N,)

    Output:
        the covariance of data of shape (d1, ..., dn, d1, ..., dn)
    """
    weights = weights / np.sum(weights)
    weighted_data = np.tensordot(np.diag(weights), data, (0, 0))

    mean_data = np.apply_along_axis(np.sum, 0, weighted_data)

    shifted_data = data - mean_data

    return np.tensordot(
        shifted_data, np.tensordot(np.diag(weights), shifted_data, (0, 0)), (0, 0)
    )


def trim_dataset(
    data: np.ndarray,
    conf_lev: float = 0.95,
    weights: Optional[list] = None,
    perturb_lev: float = 0.01,
) -> List[int]:
    """
    Remove outliers from a list of sample
    """

    data = np.array(data)
    n, n_feat = data.shape
    if (weights is None) or (np.max(weights) == np.min(weights)):
        forest = IsolationForest()
        forest.fit(data)
        scores = forest.score_samples(data)
        sorter = np.argsort(scores)

        n_remove = int((1 - conf_lev) * len(data))
        return sorter[n_remove:]
    else:
        cov_perturb = perturb_lev * get_cov(data, weights)
        choice = np.random.choice(list(range(n)), size=n, replace=True, p=weights)
        data = data[choice] + np.random.multivariate_normal(
            np.zeros(n_feat), cov_perturb, n
        )

        forest = IsolationForest()
        forest.fit(data)
        scores = forest.score_samples(data)
        sorter = np.argsort(scores)

        n_remove = int((1 - conf_lev) * len(data))
        inds = sorter[n_remove:]
        # translate inds back to original parameter
        return choice[inds]


def _safeplot_alpha(
    obj: Union[
        shapely.geometry.multipolygon.MultiPolygon, shapely.geometry.polygon.Polygon
    ],
    log: bool = False,
    contour: bool = True,
    plot=plt,
    edgecolor=None,
    **kwargs
):
    """
    Checks if object constructed through alphashape is a Polygon or a Multipolygon before plotting.

    """
    if isinstance(obj, shapely.geometry.multipolygon.MultiPolygon):
        for polygon in obj.geoms:
            X, Y = polygon.exterior.xy

            if log:
                X, Y = np.exp(X), np.exp(Y)

            if contour:
                plot.plot(X, Y, **kwargs)
            else:
                plot.fill(
                    X,
                    Y,
                    **kwargs,
                    edgecolor=edgecolor,
                )
    else:
        X, Y = obj.exterior.xy

        if log:
            X, Y = np.exp(X), np.exp(Y)

        if contour:
            plot.plot(X, Y, **kwargs)
        else:
            plot.fill(X, Y, **kwargs, edgecolor=edgecolor)
    return plot


def _plot_2D_sample_contour(
    sample,
    i,
    j,
    inds: Optional[List[int]] = None,
    weights=None,
    plot_obj=plt,
    a_shape: float = 2.0,
    log: bool = False,
    contour: bool = True,
    edgecolor=None,
    **kwargs
):
    """
    Plot 2D contours of samples.

    Contours are computed using alphashape package.

    Since alpha is used as argument name both in alphashape and matplotlib,
    the alpha of alphashape is a_shape

    If log, the data is in log space and should be plotted in normal space.

    """
    if inds is None:
        inds = list(range(len(sample)))

    if i != j:
        obj = alphashape.alphashape(sample[inds][:, [i, j]], alpha=a_shape)
        _safeplot_alpha(
            obj, plot=plot_obj, log=log, contour=contour, edgecolor=edgecolor, **kwargs
        )
    else:
        if log:
            data = np.exp(sample[:, i])
        else:
            data = sample[:, i]

        if "c" in kwargs.keys():
            if "color" in kwargs.keys():
                raise TypeError("Two arguments passed for colors ('c' and 'color')")

            color = kwargs["c"]
            kwargs.pop("c")
            kwargs["color"] = color

        sns.kdeplot(
            x=data, weights=weights, ax=plot_obj, cut=30, bw_method="scott", **kwargs
        )


def grid_plot_2D_sample_contour(
    fig,
    ax,
    sample: np.ndarray,
    conf_lev: float = 0.95,
    weights: Optional[List] = None,
    a_shape: float = 2.0,
    log: bool = False,
    **kwargs
):
    """
    Arguments:
        -fig, ax
        - sample: the sample for which contours are to be inferred
        - conf_lev: the mass which the contour should envelop
        - weights: Optional vector of weights for each point in sample
        - a_shape: alpha for alpha shape (alpha=0 -> convex envelop, alpha increases -> less regular contour)
        - log: should the shape be computed in log space ? (useful if the sample is not normalized)
    """
    data = np.array(sample)
    inds = trim_dataset(data, conf_lev=conf_lev, weights=weights)

    if log:
        if np.any(data <= 0):
            warnings.warn(
                "Some data points are non positive. Contours are computed without log transform"
            )
            log = False
        else:
            data = np.log(data)

    n = data.shape[1]

    for i in range(n):
        for j in range(n):

            _plot_2D_sample_contour(
                data,
                i=i,
                j=j,
                inds=inds,
                weights=weights,
                plot_obj=ax[j, i],
                a_shape=a_shape,
                log=log,
                **kwargs,
            )

    fig.tight_layout()

    return fig, ax
