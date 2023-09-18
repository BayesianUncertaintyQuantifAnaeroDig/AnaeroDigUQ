"""
Plotting module. Work in progress.

This submodule is not loaded directly by pyam2 or pyadm1.UQ modules (avoid dependencies)
"""

from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ...proba import Gaussian
from ...uncertainty.plot import (
    grid_plot_2D_gauss,
    grid_plot_2D_sample,
    grid_plot_2D_sample_contour,
)
from ..interp_param import default_param, par_map
from ..IO import parameter_dict
from ..proba import ref_distr


class AM2NameError(Exception):
    """Custom exception for incorrect parameter names"""


def am2_start_plot(param_names: Optional[List[str]] = None, fontsize=12):
    """
    Check that param_names are all proper parameter names
    """
    if param_names is None:
        param_names = list(parameter_dict.keys())
    elif not set(param_names).issubset(set(list(parameter_dict.keys()))):
        raise AM2NameError(
            f"""
            The following parameters are not proper AM2 parameter names:
            {set(param_names).difference(set(list(parameter_dict.keys())))}
            """
        )

    n = len(param_names)
    fig, ax = plt.subplots(n, n, figsize=(2 * n, 2 * n))

    if param_names is not None:
        for axe, name in zip(ax[0], param_names):
            axe.set_title(name, fontsize=fontsize)

        for axe, name in zip(ax[:, 0], param_names):
            axe.set_ylabel(name, fontsize=fontsize)
    return fig, ax


def _prep_data(
    sample: Union[np.ndarray, pd.DataFrame],
    weights: Optional[List[float]] = None,
    param_names: Optional[List[str]] = None,
    log: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if param_names is None:
        param_names = list(parameter_dict)

    # Remove other columns
    if isinstance(sample, pd.DataFrame):
        if (weights is None) & ("weights" in sample.columns):
            weights = sample["weights"]

        sample = np.array(sample[list(parameter_dict.keys())])

    if log:
        sample = par_map(sample)

    param_index = [parameter_dict[name] for name in param_names]
    sample = np.array(sample)[:, param_index]
    return sample, weights


def am2_plot_2D_gauss(
    fig,
    ax,
    distr: Gaussian = ref_distr,
    conf_lev: float = 0.95,
    log=True,
    param_names: Optional[List[str]] = None,
    **kwargs,
):
    """Wrap up of grid_plot_2D_gauss in AM2 context"""
    if param_names is None:
        par_index = list(range(5))
    else:
        par_index = [parameter_dict[name] for name in param_names]

    means = distr.means[par_index]
    cov = distr.cov[par_index][:, par_index]
    if log:
        means = means + np.log(default_param[par_index])
    grid_plot_2D_gauss(
        fig=fig, ax=ax, means=means, cov=cov, conf_lev=conf_lev, log=log, **kwargs
    )
    return fig, ax


def am2_plot_2D_sample(
    fig,
    ax,
    sample: Union[np.ndarray, pd.DataFrame],
    weights: Optional[List[float]] = None,
    param_names: Optional[List[str]] = None,
    log: bool = False,
    **kwargs,
):
    """
    Arguments:
        fig, ax defined through am2_start_plot
        sample: sample to plot
        weights: list of weights, optional (None -> equiweighted)
        param_names: names of parameters to plot (sample should still contain all the information)
        log: should the data be converted from log-space to normal space (through par_map)?
    """
    sample, weights = _prep_data(sample, weights, param_names=param_names, log=log)

    return grid_plot_2D_sample(fig=fig, ax=ax, sample=sample, weights=weights, **kwargs)


def am2_plot_2D_sample_contour(
    fig,
    ax,
    sample: Union[np.ndarray, pd.DataFrame],
    weights: Optional[List[float]] = None,
    conf_lev: float = 0.95,
    a_shape: float = 5.0,
    param_names: Optional[List[str]] = None,
    log: bool = False,
    **kwargs,
):
    sample, weights = _prep_data(sample, weights, param_names=param_names, log=log)

    return grid_plot_2D_sample_contour(
        fig=fig,
        ax=ax,
        sample=sample,
        conf_lev=conf_lev,
        weights=weights,
        a_shape=a_shape,
        log=True,  # Data is not normalized so log must be true to compute contours. Throws a warning if plotting in log space.
        **kwargs,
    )


def am2_plot_2D_beale_contour(
    fig,
    ax,
    sample: pd.DataFrame,
    weights: Optional[List[float]] = None,
    a_shape: float = 3.0,
    param_names: Optional[List[str]] = None,
    **kwargs,
):

    return grid_plot_2D_sample_contour(
        fig=fig,
        ax=ax,
        sample=sample[param_names],
        weights=weights,
        a_shape=a_shape,
        log=True,
        conf_lev=1.0,
        **kwargs,
    )


def am2_clean_plot(fig, ax):
    fig.tight_layout()
    return fig, ax


def am2_save_plot(fig, ax, save_path, **kwargs):
    fig.tight_layout()
    fig.savefig(save_path, **kwargs)


def am2_plot_2D_point(
    fig,
    ax,
    parameter,
    param_names: Optional[List[str]] = None,
    log: bool = False,
    marker="x",
    **kwargs,
):
    if log:
        parameter = par_map(parameter)

    n = len(param_names)
    for i, name_i in enumerate(param_names):
        for j, name_j in enumerate(param_names):
            if name_i != name_j:

                ax[j, i].plot(
                    parameter[parameter_dict[name_i]],
                    parameter[parameter_dict[name_j]],
                    marker=marker,
                    **kwargs,
                )
            else:
                ax[j, i].axvline(parameter[parameter_dict[name_i]], **kwargs)
    return fig, ax


def plotter(
    plot_prior=True,
    opt_param=None,
    param=None,
    VI=None,
    FIM=None,
    beale=None,
    MH=None,
    NB=None,
    Boot=None,
    linewidth=0.5,
    markersize=3,
    markeredgewidth=0.5,
):

    fig, ax = am2_start_plot()

    if plot_prior:
        am2_plot_2D_gauss(
            fig,
            ax,
            ref_distr,
            conf_lev=0.95,
            c="darkgreen",
            log=True,
            linewidth=linewidth,
            linestyle="--",
        )

    if opt_param is not None:
        am2_plot_2D_point(
            fig,
            ax,
            opt_param,
            color="tab:red",
            markeredgewidth=markeredgewidth,
            markersize=markersize,
            linewidth=linewidth,
        )

    if param is not None:
        am2_plot_2D_point(
            fig,
            ax,
            param,
            color="k",
            markeredgewidth=markeredgewidth,
            markersize=markersize,
            linewidth=linewidth,
        )

    if VI is not None:
        am2_plot_2D_gauss(
            fig, ax, VI, conf_lev=0.95, log=True, c="tab:green", linewidth=linewidth
        )

    if FIM is not None:
        am2_plot_2D_gauss(
            fig,
            ax,
            FIM,
            conf_lev=0.95,
            log=False,
            c="navy",
            linewidth=linewidth,
            param_names=list(parameter_dict)[:-1],
        )

    if beale is not None:
        am2_plot_2D_beale_contour(
            fig, ax, beale, c="tab:brown", a_shape=0.8, linewidth=linewidth
        )

    if MH is not None:
        am2_plot_2D_sample_contour(
            fig,
            ax,
            MH,
            conf_lev=0.95,
            c="tab:orange",
            a_shape=2.0,
            log=False,
            linewidth=linewidth,
        )

    if NB is not None:
        am2_plot_2D_sample_contour(
            fig,
            ax,
            NB,
            conf_lev=0.95,
            c="tab:olive",
            a_shape=0.8,
            log=False,
            linewidth=linewidth,
        )

    if Boot is not None:
        am2_plot_2D_sample_contour(
            fig,
            ax,
            Boot,
            conf_lev=0.95,
            c="tab:purple",
            a_shape=2.0,
            log=False,
            linewidth=linewidth,
        )

    return fig, ax


class AM2Plotter:
    """Plot manager for representation of parameter uncertainty in AM2 context"""

    def __init__(self, param_names: Optional[list[str]] = None, **kwargs):
        if param_names is None:
            param_names = list(parameter_dict)
        self.fig, self.ax = am2_start_plot(param_names, **kwargs)
        self.param_names = param_names

    def sub_names(self, param_names=None):
        if param_names is None:
            return self.param_names
        else:
            return list(set(param_names).intersection(self.param_names))

    def plot_gauss(
        self, distr=ref_distr, conf_lev=0.95, log=True, param_names=None, **kwargs
    ):
        self.fig, self.ax = am2_plot_2D_gauss(
            self.fig,
            self.ax,
            distr=distr,
            conf_lev=conf_lev,
            log=log,
            param_names=self.sub_names(param_names),
            **kwargs,
        )

    def plot_sample(self, sample, weights=None, param_names=None, **kwargs):
        self.fig, self.ax = am2_plot_2D_sample(
            self.fig,
            self.ax,
            sample=sample,
            weights=weights,
            param_names=self.sub_names(param_names),
            **kwargs,
        )

    def plot_sample_contour(
        self, sample, weights=None, a_shape=2.0, param_names=None, **kwargs
    ):
        self.fig, self.ax = am2_plot_2D_sample_contour(
            self.fig,
            self.ax,
            sample=sample,
            weights=weights,
            a_shape=a_shape,
            param_names=self.sub_names(param_names),
            **kwargs,
        )

    def plot_beale_contour(self, sample, a_shape=2.0, param_names=None, **kwargs):
        self.fig, self.ax = am2_plot_2D_beale_contour(
            self.fig,
            self.ax,
            sample=sample,
            a_shape=a_shape,
            param_names=self.sub_names(param_names),
            **kwargs,
        )

    def plot_point(
        self, parameter, param_names=None, log: bool = False, marker="x", **kwargs
    ):
        self.fig, self.ax = am2_plot_2D_point(
            self.fig,
            self.ax,
            parameter=parameter,
            param_names=self.sub_names(param_names),
            log=log,
            marker=marker,
            **kwargs,
        )

    def clean(self):
        self.fig, self.ax = am2_clean_plot(self.fig, self.ax)
