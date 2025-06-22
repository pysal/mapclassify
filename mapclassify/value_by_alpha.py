"""
functionality originally written by Stefanie Lumnitz <stefanie.lumitz@gmail.com>
as part of the ``splot`` package.

TODO:
* add Choropleth functionality with one input variable
* merge all alpha keywords in one keyword dictionary
for vba_choropleth
"""

import collections.abc

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps as cm
from matplotlib import colors, patches

from ._classify_API import classify

__author__ = "Stefanie Lumnitz <stefanie.lumitz@gmail.com>"


############################################################
#
# stuff from ``splot/_viz_value_by_alpha_mpl.py
#
############################################################


def vba_choropleth(
    x_var,
    y_var,
    gdf,
    *,
    cmap="GnBu",
    rgb=None,
    alpha=None,
    divergent=False,
    revert_alpha=False,
    ax=None,
    legend=False,
    legend_kwargs=None,
    min_alpha=0.2,
):
    """
    Value by Alpha Choropleth

    Parameters
    ----------
    x_var : string or array
        The color determining variable. It can be passed is as the name of the column
        in ``gdf`` of the actual values.
    y_var : string or array
        The alpha determining variable. It can be passed is as the name of the column
        in ``gdf`` of the actual values.
    gdf : geopandas dataframe instance
        The dataframe containing information to plot.
    cmap : str | list[str] | matplotlib.colors.Colormap
        Matplotlib colormap or list of colors used to create the vba_layer
    rgb : dict
        Keywords used for binning input values and classifying rgb values with
        ``mapclassify.classify``.  Note: valid keywords are e.g.
        ``dict(classifier='quantiles', k=5, hinge=1.5)``.
    alpha : dict
        Keywords used for binning input values and classifying alpha values with
        ``mapclassify.classify``. Note: valid keywords are e.g.
        ``dict(classifier='quantiles', k=5, hinge=1.5)``.
    divergent : bool, optional
        Creates a divergent alpha array with high values at the extremes and low,
        transparent values in the middle of the input values.
    revert_alpha : bool, optional
        If ``True``, high ``y`` values will have a low
        alpha and low values will be transparent. Default is False.
    ax : matplotlib Axes instance, optional
        Axes in which to plot the figure in multiple Axes layout. Default is None.
    legend : bool, optional
        Adds a legend. Currently, only available if data is classified
        (i.e., if ``alpha`` and ``rgb`` are used).
    legend_kwargs : dict, optional
        Keyword arguments for the legend.
    min_alpha : float = 0.2
        Minimum alpha threshold to prevent fully transparent masking.

    Returns
    -------
    fig : matplotlip Figure instance
        Figure of Value by Alpha choropleth
    ax : matplotlib Axes instance
        Axes in which the figure is plotted

    Examples
    --------

    >>> from libpysal import examples
    >>> import geopandas as gpd
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from mapclassify import vba_choropleth

    Load Example Data

    >>> link_to_data = examples.get_path('columbus.shp')
    >>> gdf = gpd.read_file(link_to_data)

    Plot a Value-by-Alpha map

    >>> fig, _ = vba_choropleth('HOVAL', 'CRIME', gdf)

    Plot a Value-by-Alpha map with reverted alpha values

    >>> fig, _ = vba_choropleth('HOVAL', 'CRIME', gdf, cmap='RdBu',
    ...                         revert_alpha=True)

    Plot a Value-by-Alpha map with classified alpha and rgb values

    >>> fig, axs = plt.subplots(2,2, figsize=(20,10))
    >>> vba_choropleth('HOVAL', 'CRIME', gdf, cmap='viridis', ax = axs[0,0],
    ...                rgb=dict(classifier='quantiles', k=3),
    ...                alpha=dict(classifier='quantiles', k=3))
    >>> vba_choropleth('HOVAL', 'CRIME', gdf, cmap='viridis', ax = axs[0,1],
    ...                rgb=dict(classifier='natural_breaks'),
    ...                alpha=dict(classifier='natural_breaks'))
    >>> vba_choropleth('HOVAL', 'CRIME', gdf, cmap='viridis', ax = axs[1,0],
    ...                rgb=dict(classifier='std_mean'),
    ...                alpha=dict(classifier='std_mean'))
    >>> vba_choropleth('HOVAL', 'CRIME', gdf, cmap='viridis', ax = axs[1,1],
    ...                rgb=dict(classifier='fisher_jenks', k=3),
    ...                alpha=dict(classifier='fisher_jenks', k=3))

    Pass in a list of colors instead of a cmap

    >>> color_list = ['#a1dab4','#41b6c4','#225ea8']
    >>> vba_choropleth('HOVAL', 'CRIME', gdf, cmap=color_list,
    ...                rgb=dict(classifier='quantiles', k=3),
    ...                alpha=dict(classifier='quantiles'))

    Add a legend and use divergent alpha values

    >>> fig = plt.figure(figsize=(15,10))
    >>> ax = fig.add_subplot(111)
    >>> vba_choropleth('HOVAL', 'CRIME', gdf, divergent=True,
    ...                alpha=dict(classifier='quantiles', k=5),
    ...                rgb=dict(classifier='quantiles', k=5),
    ...                legend=True, ax=ax,
    ...                legend_kwargs={"alpha_label": "CRIME", "rgb_label": "HOVAL"})

    """

    x = gdf[x_var].to_numpy() if isinstance(x_var, str) else x_var
    y = gdf[y_var].to_numpy() if isinstance(y_var, str) else y_var

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()

    if rgb is not None:
        rgb.setdefault("k", 5)
        rgb.setdefault("hinge", 1.5)
        rgb.setdefault("multiples", [-2, -1, 1, 2])
        rgb.setdefault("mindiff", 0)
        rgb.setdefault("initial", 100)
        rgb.setdefault("bins", [20, max(x)])
        classifier = rgb["classifier"]
        k = rgb["k"]
        hinge = rgb["hinge"]
        multiples = rgb["multiples"]
        mindiff = rgb["mindiff"]
        initial = rgb["initial"]
        bins = rgb["bins"]
        rgb_bins = classify(
            x,
            classifier,
            k=k,
            hinge=hinge,
            multiples=multiples,
            mindiff=mindiff,
            initial=initial,
            bins=bins,
        )
        x = rgb_bins.yb

    if alpha is not None:
        alpha.setdefault("k", 5)
        alpha.setdefault("hinge", 1.5)
        alpha.setdefault("multiples", [-2, -1, 1, 2])
        alpha.setdefault("mindiff", 0)
        alpha.setdefault("initial", 100)
        alpha.setdefault("bins", [20, max(y)])
        classifier = alpha["classifier"]
        k = alpha["k"]
        hinge = alpha["hinge"]
        multiples = alpha["multiples"]
        mindiff = alpha["mindiff"]
        initial = alpha["initial"]
        bins = alpha["bins"]
        # TODO: use the pct keyword here
        alpha_bins = classify(
            y,
            classifier,
            k=k,
            hinge=hinge,
            multiples=multiples,
            mindiff=mindiff,
            initial=initial,
            bins=bins,
        )
        y = alpha_bins.yb

    rgba, vba_cmap = _value_by_alpha_cmap(
        x,
        y,
        cmap=cmap,
        divergent=divergent,
        revert_alpha=revert_alpha,
        min_alpha=min_alpha,
    )
    gdf.plot(color=rgba, ax=ax)
    ax.set_axis_off()
    ax.set_aspect("equal")

    if legend:
        left, bottom, width, height = [0, 0.5, 0.2, 0.2]
        ax2 = fig.add_axes([left, bottom, width, height])
        if legend_kwargs:
            legend_kwargs |= {"ax": ax2}
        else:
            legend_kwargs = {"ax": ax2}
        legend_kwargs |= {"min_alpha": min_alpha}
        _vba_legend(rgb_bins, alpha_bins, vba_cmap, **legend_kwargs)
    return fig, ax


def _value_by_alpha_cmap(
    x,
    y,
    *,
    cmap="GnBu",
    divergent=False,
    revert_alpha=False,
    min_alpha=0.2,
):
    """Calculates Value by Alpha rgba values

    Parameters
    ----------
    x : array
        The color determining variable values.
    y : array
        The alpha determining variable values.
    cmap : str | list[str] | matplotlib.colors.Colormap
        Matplotlib colormap or list of colors used to create the vba_layer
    divergent : bool, optional
        Creates a divergent alpha array with high values at the extremes and low,
        transparent values in the middle of the input values.
    revert_alpha : bool, optional
        If ``True``, high ``y`` values will have a low
        alpha and low values will be transparent. Default is False.
    min_alpha : float = 0.2
        Minimum alpha threshold to prevent fully transparent masking.

    Returns
    -------
    rgba : ndarray (n,4)
        RGBA colormap, where the alpha channel represents one
        attribute (x) and the rgb color the other attribute (y)
    cmap : matplotlib.colors.Colormap
        Colormap for the VBA layer

    """
    # option for cmap or colorlist input
    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)
    elif isinstance(cmap, collections.abc.Sequence):
        cmap = colors.LinearSegmentedColormap.from_list("newmap", cmap)

    # normalize rgb (x) and alpha (y)
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = min_alpha + ((y - y.min()) / (y.max() - y.min()) * (1 - min_alpha))

    rgba = cmap(x_norm)
    if revert_alpha:
        rgba[:, 3] = 1 - y_norm
    else:
        rgba[:, 3] = y_norm

    if divergent is not False:
        a_under_0p5 = rgba[:, 3] < 0.5
        rgba[a_under_0p5, 3] = 1 - rgba[a_under_0p5, 3]
        rgba[:, 3] = (rgba[:, 3] - 0.5) * 2
    return rgba, cmap


def _vba_legend(
    rgb_bins,
    alpha_bins,
    cmap,
    *,
    ax=None,
    rgb_label=None,
    alpha_label=None,
    min_alpha=0.2,
):
    """Creates Value by Alpha heatmap used as choropleth legend.

    Parameters
    ----------
    rgb_bins : mapclassify instance
        Object of classified values used for rgb.
    alpha_bins : mapclassify instance
        Object of classified values used for alpha.
    cmap : matplotlib.colors.Colormap
        Colormap for the VBA layer
    ax : matplotlib Axes instance, optional
        Axes in which to plot the figure in multiple Axes layout. Default is None
    rgb_label : str, optional
        Label for the y-axis; the rgb variable.
    alpha_label : str, optional
        Label for the x-axis; the alpha variable.
    min_alpha : float = 0.2
        Minimum alpha threshold to prevent fully transparent masking.

    Returns
    -------
    fig : matplotlip Figure instance
        Figure of Value by Alpha heatmap
    ax : matplotlib Axes instance
        Axes in which the figure is plotted

    """
    # VALUES
    rgba, legend_cmap = _value_by_alpha_cmap(
        rgb_bins.yb, alpha_bins.yb, cmap=cmap, min_alpha=min_alpha
    )
    # separate rgb and alpha values
    alpha = rgba[:, 3]
    # extract unique values for alpha and rgb
    alpha_vals = np.unique(alpha)
    rgb_vals = legend_cmap(
        (rgb_bins.bins - rgb_bins.bins.min())
        / (rgb_bins.bins.max() - rgb_bins.bins.min())
    )[:, 0:3]

    # PLOTTING
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()

    for irow, alpha_val in enumerate(alpha_vals):
        for icol, rgb_val in enumerate(rgb_vals):
            rect = patches.Rectangle(
                (irow, icol),
                1,
                1,
                linewidth=3,
                edgecolor="none",
                facecolor=rgb_val,
                alpha=alpha_val,
            )
            ax.add_patch(rect)

    values_alpha, x_in_thousand = format_legend(alpha_bins.bins)
    values_rgb, y_in_thousand = format_legend(rgb_bins.bins)
    ax.plot([], [])
    ax.set_xlim([0, irow + 1])
    ax.set_ylim([0, icol + 1])
    ax.set_xticks(np.arange(irow + 1) + 0.5)
    ax.set_yticks(np.arange(icol + 1) + 0.5)
    ax.set_xticklabels(
        [f"< {val:1.1f}" for val in values_alpha],
        rotation=30,
        horizontalalignment="right",
    )
    ax.set_yticklabels([f"$<${val:1.1f}" for val in values_rgb])

    alpha_label = alpha_label if alpha_label else "alpha variable"
    rgb_label = rgb_label if rgb_label else "rgb variable"

    if x_in_thousand:
        ax.set_xlabel(f"{alpha_label} ($10^3$)")
    if y_in_thousand:
        ax.set_ylabel(f"{rgb_label} ($10^3$)")
    else:
        ax.set_xlabel(alpha_label)
        ax.set_ylabel(rgb_label)

    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)
    return fig, ax


############################################################
#
# stuff from ``splot/_viz_utils.py
#
############################################################


def format_legend(values):
    """Helper to return sensible legend values

    Parameters
    ----------
    values : array
        Values plotted in legend.
    """
    in_thousand = False
    if np.any(values > 1000):
        in_thousand = True
        values = values / 1000
    return values, in_thousand


# Utility function #1 - forces continuous diverging colormap to be centered at zero
def shift_colormap(cmap, *, start=0, midpoint=0.5, stop=1.0, name="shiftedcmap"):
    """Offset the "center" of a colormap. Useful for data with a negative min and
    positive max and you want the middle of the colormap's dynamic range to be at zero.

    Parameters
    ----------
    cmap : str or matplotlib.colors.Colormap
        Colormap to be altered
    start : float, optional
        Offset from lowest point in the colormap's range. Should be between 0.0
        and ``midpoint``. Default is 0.0 (no lower ofset).
    midpoint : float, optional
        The new center of the colormap. Should be between 0.0 and 1.0. In general, this
        should be ``1 - vmax/(vmax + abs(vmin))``. For example, if your data range from
        -15.0 to +5.0 and you want the center of the colormap at 0.0, ``midpoint``
        should be set to  1 - 5/(5 + 15)) or 0.75. Default is 0.5 (no shift).
    stop : float, optional
        Offset from highest point in the colormap's range. Should be between
        ``midpoint`` and 1.0. Default is 1.0 (no upper ofset).
    name : str, optional
        Name of the new colormap.

    Returns
    -------
    new_cmap : matplotlib.colors.Colormap
        A new colormap that has been shifted.
    """
    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)

    cdict = {"red": [], "green": [], "blue": [], "alpha": []}

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack(
        [
            np.linspace(0.0, midpoint, 128, endpoint=False),
            np.linspace(midpoint, 1.0, 129, endpoint=True),
        ]
    )

    for ri, si in zip(reg_index, shift_index, strict=True):
        r, g, b, a = cmap(ri)

        cdict["red"].append((si, r, r))
        cdict["green"].append((si, g, g))
        cdict["blue"].append((si, b, b))
        cdict["alpha"].append((si, a, a))

    new_cmap = colors.LinearSegmentedColormap(name, cdict)
    cm.register(new_cmap)
    return new_cmap


# Utility #2 - truncate colorcap in order to grab only positive or negative portion
def truncate_colormap(cmap, *, minval=0.0, maxval=1.0, n=100):
    """Truncate a colormap by selecting a subset of the original colormap's values.

    Parameters
    ----------
    cmap : str or matplotlib.colors.Colormap
        Colormap to be altered
    minval : float, optional
        Minimum value of the original colormap to include
        in the truncated colormap. Default is 0.0.
    maxval : float, optional
        Maximum value of the original colormap to
        include in the truncated colormap. Default is 1.0.
    n : int, optional
        Number of intervals between the min and max values
        for the gradient of the truncated colormap. Default is 100.

    Returns
    -------
    new_cmap : matplotlib.colors.Colormap
        A new colormap that has been shifted.
    """

    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)

    new_cmap = colors.LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap
