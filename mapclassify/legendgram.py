import warnings

import numpy as np


def _legendgram(
    classifier,
    *,
    ax=None,
    cmap=None,
    bins=50,
    inset=True,
    clip=None,
    vlines=False,
    vlinecolor="black",
    vlinewidth=1,
    loc="lower left",
    legend_size=("27%", "20%"),
    frameon=False,
    tick_params=None,
    bbox_to_anchor=None,
    **kwargs,
):
    """See ``classifiers.MapClassifier.plot_legendgram()`` docstring."""

    try:
        import matplotlib.pyplot as plt
        from matplotlib import colormaps
        from matplotlib.collections import Collection
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    except ImportError as e:
        raise ImportError from e("you must have matplotlib ")

    def _get_cmap(_ax):
        """Detect the most recent matplotlib colormap used, if previously rendered."""

        _child_cmaps = [
            (cc.cmap, cc.cmap.name)
            for cc in _ax.properties()["children"]
            if isinstance(cc, Collection)
        ]
        has_child_cmaps = len(_child_cmaps)
        n_unique_cmaps = len({cc[1] for cc in _child_cmaps})
        if has_child_cmaps:
            _cmap, cmap_name = _child_cmaps[-1]
            if n_unique_cmaps > 1:
                warnings.warn(
                    (
                        f"There are {n_unique_cmaps} unique colormaps associated with "
                        f"the axes. Defaulting to most recent colormap: '{cmap_name}'"
                    ),
                    UserWarning,
                    stacklevel=2,
                )
        else:
            warnings.warn(
                "There is no data associated with the `ax`.", UserWarning, stacklevel=2
            )
            _cmap = colormaps.get_cmap("viridis")
        return _cmap

    if ax is None:
        f, ax = plt.subplots()
    else:
        f = ax.get_figure()

    if cmap is None:
        cmap = _get_cmap(ax)
    if isinstance(cmap, str):
        cmap = colormaps[cmap]

    k = len(classifier.bins)
    breaks = classifier.bins

    if inset:
        if not bbox_to_anchor:
            bbox_to_anchor = (0, 0, 1, 1)
        histpos = inset_axes(
            ax,
            loc=loc,
            width=legend_size[0],
            height=legend_size[1],
            bbox_to_anchor=bbox_to_anchor,
            bbox_transform=ax.transAxes,
        )
        histax = f.add_axes(histpos)
    else:
        histax = ax
    _, bins, patches = histax.hist(classifier.y, bins=bins, color="0.1", **kwargs)

    colors = [cmap(i) for i in np.linspace(0, 1, k)]

    bucket_breaks = [0] + [np.searchsorted(bins, i) for i in breaks]
    for c in range(k):
        for b in range(bucket_breaks[c], bucket_breaks[c + 1]):
            patches[b].set_facecolor(colors[c])
    if clip is not None:
        histax.set_xlim(*clip)
    histax.set_frame_on(frameon)
    histax.get_yaxis().set_visible(False)
    if tick_params is None:
        tick_params = {}
    if vlines:
        lim = histax.get_ylim()[1]
        # plot upper limit of each bin
        for i in classifier.bins:
            histax.vlines(i, 0, lim, color=vlinecolor, linewidth=vlinewidth)
    tick_params["labelsize"] = tick_params.get("labelsize", 12)
    histax.tick_params(**tick_params)
    return histax
