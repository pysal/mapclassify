"""
functionality originally written by Stefanie Lumnitz <stefanie.lumitz@gmail.com>
as part of the ``splot`` package.

TODO:
* add Choropleth functionality with one input variable
* merge all alpha keywords in one keyword dictionary
for vba_choropleth
"""

import collections.abc

import numpy as np

from ._classify_API import classify
from .classifiers import _format_intervals

try:
    import matplotlib.pyplot as plt
    from matplotlib import colormaps as cm
    from matplotlib import colors, patches

    MPL_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    MPL_AVAILABLE = False

MPL_NOT_AVAILABLE = ImportError("you must have matplotlib")


__author__ = "Stefanie Lumnitz <stefanie.lumitz@gmail.com>"


############################################################
#
# stuff from ``splot/_viz_value_by_alpha_mpl.py
#
############################################################


def vba_choropleth(
    x,
    y,
    gdf,
    *,
    cmap="GnBu",
    x_classification_kwds=None,
    y_classification_kwds=None,
    divergent=False,
    revert_alpha=False,
    ax=None,
    legend=False,
    legend_kwargs=None,
    min_alpha=0.2,
):
    """Generate Value by Alpha Choropleth plots.

    A Value-by-Alpha Choropleth is a bivariate choropleth that uses the values
    of the second input variable ``y`` as a transparency mask, determining how much
    of the choropleth displaying the values of a first variable ``x`` is shown.

    Parameters
    ----------
    x : str | numpy.ndarray | pandas.Series
        The color determining variable. It can be passed is as the name of the column
        in ``gdf`` or the actual values.
    y : str | numpy.ndarray | pandas.Series
        The alpha determining variable. It can be passed is as the name of the column
        in ``gdf`` or the actual values.
    gdf : geopandas dataframe instance
        The dataframe containing information to plot.
    cmap : str | list[str] | matplotlib.colors.Colormap
        Matplotlib colormap or list of colors used to create the vba_layer
    x_classification_kwds : dict
        Keywords used for binning and classifying color-determinant input values with
        ``mapclassify.classify``.
    y_classification_kwds : dict
        Keywords used for binning and classifying alpha-determinant input values with
        ``mapclassify.classify``.
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
    >>> plt.show()  # doctest: +SKIP
    >>> plt.close()

    Plot a Value-by-Alpha map with reverted alpha values

    >>> fig, _ = vba_choropleth('HOVAL', 'CRIME', gdf, cmap='RdBu',
    ...                         revert_alpha=True)
    >>> plt.show()  # doctest: +SKIP
    >>> plt.close()

    Plot a Value-by-Alpha map with classified alpha and rgb values

    >>> fig, axs = plt.subplots(2,2, figsize=(20,10))
    >>> vba_choropleth('HOVAL', 'CRIME', gdf, cmap='viridis', ax = axs[0,0],
    ...                x_classification_kwds=dict(classifier='quantiles', k=3),
    ...                y_classification_kwds=dict(classifier='quantiles', k=3))  # doctest: +SKIP
    >>> vba_choropleth('HOVAL', 'CRIME', gdf, cmap='viridis', ax = axs[0,1],
    ...                x_classification_kwds=dict(classifier='natural_breaks'),
    ...                y_classification_kwds=dict(classifier='natural_breaks'))  # doctest: +SKIP
    >>> vba_choropleth('HOVAL', 'CRIME', gdf, cmap='viridis', ax = axs[1,0],
    ...                x_classification_kwds=dict(classifier='std_mean'),
    ...                y_classification_kwds=dict(classifier='std_mean'))  # doctest: +SKIP
    >>> vba_choropleth('HOVAL', 'CRIME', gdf, cmap='viridis', ax = axs[1,1],
    ...                x_classification_kwds=dict(classifier='fisher_jenks', k=3),
    ...                y_classification_kwds=dict(classifier='fisher_jenks', k=3))  # doctest: +SKIP
    >>> plt.show()  # doctest: +SKIP
    >>> plt.close()

    Pass in a list of colors instead of a cmap

    >>> color_list = ['#a1dab4','#41b6c4','#225ea8']
    >>> vba_choropleth('HOVAL', 'CRIME', gdf, cmap=color_list,
    ...                x_classification_kwds=dict(classifier='quantiles', k=3),
    ...                y_classification_kwds=dict(classifier='quantiles'))  # doctest: +SKIP
    >>> plt.show()  # doctest: +SKIP
    >>> plt.close()

    Add a legend and use divergent alpha values

    >>> fig = plt.figure(figsize=(15,10))
    >>> ax = fig.add_subplot(111)
    >>> vba_choropleth('HOVAL', 'CRIME', gdf, divergent=True,
    ...                x_classification_kwds=dict(classifier='quantiles', k=5),
    ...                y_classification_kwds=dict(classifier='quantiles', k=5),
    ...                legend=True, ax=ax,
    ...                legend_kwargs={"alpha_label": "CRIME", "rgb_label": "HOVAL"})  # doctest: +SKIP
    >>> plt.show()  # doctest: +SKIP
    >>> plt.close()

    """  # noqa: E501

    if not MPL_AVAILABLE:
        raise MPL_NOT_AVAILABLE

    if legend and (not x_classification_kwds or not y_classification_kwds):
        raise ValueError(
            "Plotting a legend requires classification for both the `x` and `y` "
            "variables. See `x_classification_kwds` and `y_classification_kwds`."
        )

    x = gdf[x].to_numpy() if isinstance(x, str) else x
    y = gdf[y].to_numpy() if isinstance(y, str) else y

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()

    if x_classification_kwds is not None:
        classifier = x_classification_kwds.pop("classifier")
        x_bins = classify(x, classifier, **x_classification_kwds)
        x = x_bins.yb

    if y_classification_kwds is not None:
        classifier = y_classification_kwds.pop("classifier")
        # TODO: use the pct keyword here
        y_bins = classify(y, classifier, **y_classification_kwds)
        y = y_bins.yb

    rgba, vba_cmap = _value_by_alpha_cmap(
        x,
        y,
        cmap=cmap,
        divergent=divergent,
        revert_alpha=revert_alpha,
        min_alpha=min_alpha,
    )
    gdf.plot(color=rgba, ax=ax)

    if legend:
        left, bottom, width, height = [0, 0.5, 0.2, 0.2]
        ax2 = fig.add_axes([left, bottom, width, height])
        if legend_kwargs:
            legend_kwargs |= {"ax": ax2}
        else:
            legend_kwargs = {"ax": ax2}
        legend_kwargs |= {"min_alpha": min_alpha}
        _vba_legend(x_bins, y_bins, vba_cmap, **legend_kwargs)
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

    if not MPL_AVAILABLE:
        raise MPL_NOT_AVAILABLE

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
    x_bins,
    y_bins,
    cmap,
    *,
    ax=None,
    x_label=None,
    y_label=None,
    min_alpha=0.2,
):
    """Creates Value by Alpha heatmap used as choropleth legend.

    Parameters
    ----------
    x_bins : MapClassifier
        Object of classified values used for color.
    y_bins : MapClassifier
        Object of classified values used for alpha.
    cmap : matplotlib.colors.Colormap
        Colormap for the VBA layer
    ax : matplotlib Axes instance, optional
        Axes in which to plot the figure in multiple Axes layout. Default is None
    x_label : str, optional
        Label for the x-axis; the color variable.
    y_label : str, optional
        Label for the y-axis; the alpha variable.
    min_alpha : float = 0.2
        Minimum alpha threshold to prevent fully transparent masking.

    Returns
    -------
    fig : matplotlip Figure instance
        Figure of Value by Alpha heatmap
    ax : matplotlib Axes instance
        Axes in which the figure is plotted
    """

    if not MPL_AVAILABLE:
        raise MPL_NOT_AVAILABLE

    # VALUES
    rgba, legend_cmap = _value_by_alpha_cmap(
        x_bins.yb, y_bins.yb, cmap=cmap, min_alpha=min_alpha
    )
    # separate rgb and alpha values
    alpha = rgba[:, 3]
    # extract unique values for alpha and rgb
    alpha_vals = np.unique(alpha)
    rgb_vals = legend_cmap(
        (x_bins.bins - x_bins.bins.min()) / (x_bins.bins.max() - x_bins.bins.min())
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

    values_rgb, _, x_in_thousand = _format_intervals(x_bins, fmt="{:.1f}")
    values_alpha, _, y_in_thousand = _format_intervals(y_bins, fmt="{:.1f}")

    ax.plot([], [])
    ax.set_xlim([0, irow + 1])
    ax.set_ylim([0, icol + 1])
    ax.set_xticks(np.arange(irow + 1) + 0.5)
    ax.set_yticks(np.arange(icol + 1) + 0.5)
    ax.set_xticklabels(
        [f"$<${val}" for val in values_rgb[1:]],
        rotation=30,
        horizontalalignment="right",
    )
    ax.set_yticklabels([f"$<${val}" for val in values_alpha[1:]])

    x_label = x_label if x_label else "x-rgb variable"
    y_label = y_label if y_label else "y-alpha variable"

    if x_in_thousand:
        ax.set_xlabel(f"{x_label} ($10^3$)")
    if y_in_thousand:
        ax.set_ylabel(f"{y_label} ($10^3$)")
    else:
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

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
    if not MPL_AVAILABLE:
        raise MPL_NOT_AVAILABLE

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

    if not MPL_AVAILABLE:
        raise MPL_NOT_AVAILABLE

    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)

    new_cmap = colors.LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap
