"""
Module for plotting a point in a grid plot.
"""


def plot_2D_point(
    fig,
    ax,
    parameter,
    marker="x",
    **kwargs,
):
    n = len(parameter)
    for i in range(n):
        for j in range(n):
            if i != j:

                ax[j, i].plot(
                    parameter[i],
                    parameter[j],
                    marker=marker,
                    **kwargs,
                )
            else:
                ax[j, i].axvline(parameter[i], **kwargs)
    return fig, ax
