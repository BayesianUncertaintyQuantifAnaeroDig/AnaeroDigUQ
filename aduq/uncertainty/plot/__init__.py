""" Module for plotting representations of confidence regions.
Work in progress.
"""

from .gauss import gaussian_ellipse, grid_plot_2D_gauss
from .init_plot import start_plot
from .point import plot_2D_point
from .sample import grid_plot_2D_sample
from .sample_contour import _safeplot_alpha, grid_plot_2D_sample_contour, trim_dataset
