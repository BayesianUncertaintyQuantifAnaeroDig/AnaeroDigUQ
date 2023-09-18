"""
Plotting module. Work in progress.

This submodule is not loaded directly by pyadm1 or pyadm1.UQ modules.

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
from ...uncertainty.plot.sample_contour import _plot_2D_sample_contour
from ..IO import free_to_param, parameter_dict
from ..IO._normalisation_param import renorm_param
from ..proba import ref_distr


def adm1_start_plot(param_names: Optional[List[str]] = None, fontsize=12):
    """
    Check that param_names are all proper parameter names
    """
    if param_names is None:
        param_names = list(parameter_dict.keys())
    elif not set(param_names).issubset(set(list(parameter_dict.keys()))):
        raise NameError(
            f"""
            The following parameters are not proper ADM1 parameter names:
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
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if param_names is None:
        param_names = list(parameter_dict)

    # Remove other columns
    if isinstance(sample, pd.DataFrame):
        if (weights is None) & ("weights" in sample.columns):
            weights = sample["weights"]

        sample = np.array(sample[list(parameter_dict.keys())])

    param_index = [parameter_dict[name] for name in param_names]
    sample = np.array(sample)[:, param_index]
    return sample, weights


def adm1_plot_2D_gauss(
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
        print(param_names)
        par_index = [parameter_dict[name] for name in param_names]

    means = distr.means[par_index]
    cov = distr.cov[par_index][:, par_index]
    if log:
        means = means + np.log([renorm_param[name] for name in param_names])
    return grid_plot_2D_gauss(
        fig=fig, ax=ax, means=means, cov=cov, conf_lev=conf_lev, log=log, **kwargs
    )


def adm1_plot_2D_sample(
    fig,
    ax,
    sample: Union[np.ndarray, pd.DataFrame],
    weights: Optional[List[float]] = None,
    param_names: Optional[List[str]] = None,
    **kwargs,
):
    """
    Arguments:
        fig, ax defined through AM2_start_plot
        sample: sample to plot
        weights: list of weights, optional (None -> equiweighted)
        param_names: names of parameters to plot (sample should still contain all the information)
    """
    sample, weights = _prep_data(sample, weights, param_names=param_names)

    return grid_plot_2D_sample(fig=fig, ax=ax, sample=sample, weights=weights, **kwargs)


def adm1_plot_2D_sample_contour(
    fig,
    ax,
    sample: Union[np.ndarray, pd.DataFrame],
    weights: Optional[List[float]] = None,
    a_shape: float = 5.0,
    param_names: Optional[List[str]] = None,
    log: bool = False,
    **kwargs,
):

    sample, weights = _prep_data(sample, weights, param_names=param_names)

    return grid_plot_2D_sample_contour(
        fig=fig,
        ax=ax,
        sample=sample,
        weights=weights,
        a_shape=a_shape,
        log=True,  # Data is not normalized so log must be true to compute contours. Throws a warning if plotting in log space.
        **kwargs,
    )


def adm1_plot_2D_beale_contour(
    fig,
    ax,
    sample: pd.DataFrame,
    param_names: Optional[List[str]] = None,
    a_shape: float = 3.0,
    **kwargs,
):
    if param_names is None:
        param_names = list(parameter_dict)

    param_names_loc = set(param_names).intersection(sample.columns)
    param_index = [i for i, name in enumerate(param_names) if name in param_names_loc]

    # Convert to log
    data = np.log(np.array(sample))

    for i in param_index:
        for j in param_index:
            if i != j:
                _plot_2D_sample_contour(
                    data,
                    i=i,
                    j=j,
                    plot_obj=ax[j, i],
                    a_shape=a_shape,
                    log=True,
                    **kwargs,
                )

    fig.tight_layout()

    return fig, ax


def adm1_save_plot(fig, ax, save_path, **kwargs):
    fig.tight_layout()
    fig.savefig(save_path, **kwargs)


def adm1_plot_2D_point(
    fig,
    ax,
    parameter,
    param_names: Optional[List[str]] = None,
    log: bool = False,
    marker="x",
    **kwargs,
):
    if log:
        parameter = np.array(free_to_param(parameter))

    if param_names is None:
        param_names = list(parameter_dict)

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


class adm1_plotter:
    def __init__(self, param_names: Optional[list[str]] = None):
        if param_names is None:
            param_names = list(parameter_dict)
        self.fig, self.ax = adm1_start_plot(param_names)
        self.param_names = param_names

    def sub_names(self, param_names=None):
        if param_names is None:
            return self.param_names
        else:
            return list(set(param_names).intersection(self.param_names))

    def plot_gauss(
        self, distr=ref_distr, conf_lev=0.95, log=True, param_names=None, **kwargs
    ):
        self.fig, self.ax = adm1_plot_2D_gauss(
            self.fig,
            self.ax,
            distr=distr,
            conf_lev=conf_lev,
            log=log,
            param_names=self.sub_names(param_names),
            **kwargs,
        )

    def plot_sample(self, sample, weights=None, param_names=None, **kwargs):
        self.fig, self.ax = adm1_plot_2D_sample(
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
        self.fig, self.ax = adm1_plot_2D_sample_contour(
            self.fig,
            self.ax,
            sample=sample,
            weights=weights,
            a_shape=a_shape,
            param_names=self.sub_names(param_names),
            **kwargs,
        )

    def plot_beale_contour(self, sample, a_shape=2.0, param_names=None, **kwargs):
        self.fig, self.ax = adm1_plot_2D_beale_contour(
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
        self.fig, self.ax = adm1_plot_2D_point(
            self.fig,
            self.ax,
            parameter=parameter,
            param_names=self.sub_names(param_names),
            log=log,
            marker=marker,
            **kwargs,
        )
