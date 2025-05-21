import numpy as np


def _legendgram(
    classifier,
    *,
    ax=None,
    cmap="viridis",
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
):
    """
    Add a histogram in a choropleth with colors aligned with map ...

    Arguments
    ---------
    ax : Axes
    ...
    loc : string or int
        valid legend location like that used in matplotlib.pyplot.legend. Valid
        locations are 'upper left', 'upper center', 'upper right', 'center left',
        'center', 'center right', 'lower left', 'lower center', 'lower right'.
    legend_size : tuple
        tuple of floats or strings describing the (width, height) of the
        legend. If a float is provided, it is
        the size in inches, e.g. ``(1.3, 1)``. If a string is provided, it is
        the size in relative units, e.g. ``('40%', '20%')``. By default,
        i.e. if ``bbox_to_anchor`` is not specified, those are relative to
        the `ax`. Otherwise, they are to be understood relative to the
        bounding box provided via ``bbox_to_anchor``.
    frameon : bool (default: False)
        whether to add a frame to the legendgram
    tick_params : keyword dictionary
        options to control how the histogram axis gets ticked/labelled.
    bbox_to_anchor : tuple or ``matplotlib.trasforms.BboxBase``
        Bbox that the inset axes will be anchored to. If None, a tuple of
        ``(0, 0, 1, 1)`` is used. If a tuple, can be either
        ``[left, bottom, width, height]``, or ``[left, bottom]``. If the ``legend_size``
        is in relative units (%), the 2-tuple ``[left, bottom]`` cannot be used.

    Returns
    -------
    axis containing the legendgram.
    """

    try:
        import matplotlib.pyplot as plt
        from matplotlib import colormaps
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    except ImportError as e:
        raise ImportError from e("you must have matplotlib ")
    if ax is None:
        f, ax = plt.subplots()
    else:
        f = ax.get_figure()
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
    N, bins, patches = histax.hist(classifier.y, bins=bins, color="0.1")
    pl = colormaps[cmap]

    colors = [pl(i) for i in np.linspace(0, 1, k)]

    bucket_breaks = [0] + [np.searchsorted(bins, i) for i in breaks]
    for c in range(k):
        for b in range(bucket_breaks[c], bucket_breaks[c + 1]):
            patches[b].set_facecolor(colors[c])
    if clip is not None:
        histax.set_xlim(*clip)
    histax.set_frame_on(frameon)
    histax.get_yaxis().set_visible(False)
    if tick_params is None:
        tick_params = dict()
    if vlines:
        lim = histax.get_ylim()[1]
        # plot upper limit of each bin
        for i in classifier.bins:
            histax.vlines(i, 0, lim, color=vlinecolor, linewidth=vlinewidth)
    tick_params["labelsize"] = tick_params.get("labelsize", 12)
    histax.tick_params(**tick_params)
    return histax
