from math import pi

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2, lognorm, norm


def gaussian_ellipse(
    mean: np.ndarray,
    cov: np.ndarray,
    conf_lev: float = 0.95,
    log=False,
    n_points=400,
    plot=plt,
    contour=True,
    edgecolor=None,
    *args,
    **kwargs
):
    """
    Representation of a 2D gaussian distribution as an ellipse (smallest confidence region)

    Args:
        mean: mean of the gaussian distribution
        cov: covariance the gaussian distribution
        conf_lev: confidence level required for the ellipse (default: 0.95)
        log: Is the distribution log gaussian rather than gaussian (default is False)?
        n_points: number of points used to represent the ellipse boundary
        plot: where the plot should be drawn (default matplotlib.pyplot)
        contour: Should the ellipse contour be drawn or should it be filled (default is False)?

    Further args, kwargs are passed to either plot (contour =True) or fill function of matplotlib

    """

    vals, vects = np.linalg.eigh(cov)
    cov12 = (np.sqrt(vals) * vects.T) @ vects
    angles = np.linspace(0, 2 * pi, n_points)
    circ = np.sqrt(chi2(2).ppf(conf_lev)) * np.array(
        [np.cos(angles), np.sin(angles)]
    )  # 2, N
    ellipse = cov12 @ circ
    X = mean[0] + ellipse[0]
    Y = mean[1] + ellipse[1]
    if log:
        X = np.exp(X)
        Y = np.exp(Y)
    if contour:
        plot.plot(X, Y, *args, **kwargs)
    else:
        plot.fill(X, Y, *args, **kwargs, edgecolor=edgecolor)


def _plot_2D_gauss(
    mean,
    cov,
    conf_lev,
    i,
    j,
    log=False,
    plot_obj=plt,
    n_points=400,
    contour=True,
    edgecolor=None,
    **kwargs
):
    """
    Helper function to draw multi dimensional gaussian distributions as a matrix of 2D plots.
    Dispatch between two draw mode:
    - either an ellipse if i != j,
    - marginal density if i==j.

    """
    if i != j:
        gaussian_ellipse(
            mean[[i, j]],
            cov[[i, j]][:, [i, j]],
            conf_lev=conf_lev,
            log=log,
            plot=plot_obj,
            n_points=n_points,
            contour=contour,
            edgecolor=edgecolor,
            **kwargs
        )
    else:
        mu, sigma = mean[i], np.sqrt(cov[i, i])
        x = np.linspace(mu - 6 * sigma, mu + 6 * sigma, n_points)
        if log:
            plot_obj.plot(
                np.exp(x), lognorm(s=sigma, scale=np.exp(mu)).pdf(np.exp(x)), **kwargs
            )
        else:
            plot_obj.plot(x, norm(mu, sigma).pdf(x), **kwargs)
    return plot_obj


def grid_plot_2D_gauss(
    fig,
    ax,
    means: np.ndarray,
    cov: np.ndarray = None,
    conf_lev: float = 0.95,
    log: bool = False,
    contour: bool = True,
    **kwargs
):
    """
    Represent multi dimensional gaussian distribution as a matrix of 2D plots.
    Draw either an ellipse if i != j, or the marginal density if i==j.

    Args:
        fig, ax: the plot on which the gaussian distribution should be represented.
        means: the means of the gaussian distribution
        cov: the covariance of the gaussian distribution
        conf_lev: confidence level required for the ellipse (default: 0.95)
        log: Is the distribution log gaussian rather than gaussian (default is False)?
        contour: Should the ellipse contour be drawn or should it be filled (default is False)?
    Further *args and **kwargs are passed to plt.fill and plt.plot.
    """

    n = len(means)

    for i in range(n):
        for j in range(n):

            _plot_2D_gauss(
                mean=means,
                cov=cov,
                conf_lev=conf_lev,
                i=i,
                j=j,
                log=log,
                plot_obj=ax[j, i],
                contour=contour,
                **kwargs
            )

    return fig, ax
