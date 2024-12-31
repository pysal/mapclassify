import numpy as np


def _legendgram(
    classifier,
    ax=None,
    cmap="viridis",
    bins=50,
    inset=True,
    clip=None,
    vlines=False,
    vlinecolor='black',
    vlinewidth=1,
    loc="lower left",
    legend_size=(0.27, 0.2),
    frameon=False,
    tick_params=None,
):
    """
    Add a histogram in a choropleth with colors aligned with map
    ...

    Arguments
    ---------
    f           : Figure
    ax          : AxesSubplot
    y           : ndarray/Series
                  Values to map
    breaks      : list
                  Sequence with breaks for each class (i.e. boundary values
                  for colors)
    pal         : palettable colormap or matplotlib colormap
    clip        : tuple
                  [Optional. Default=None] If a tuple, clips the X
                  axis of the histogram to the bounds provided.
    loc         :   string or int
                    valid legend location like that used in matplotlib.pyplot.legend
    legend_size : tuple
                  tuple of floats between 0 and 1 describing the (width,height)
                  of the legend relative to the original frame.
    frameon     : bool (default: False)
                  whether to add a frame to the legendgram
    tick_params : keyword dictionary
                  options to control how the histogram axis gets ticked/labelled.

    Returns
    -------
    axis containing the legendgram.
    """

    try:
        import matplotlib.pyplot as plt
        from matplotlib import colormaps
    except ImportError as e:
        raise ImportError from e("you must have matplotlib ")
    if ax is None:
        f, ax = plt.subplots()
    else:
        f = ax.get_figure()
    k = len(classifier.bins)
    breaks = classifier.bins
    if inset:
        histpos = _make_location(ax, loc, legend_size=legend_size)
        histax = f.add_axes(histpos)
    else:
        histax = ax
    N, bins, patches = histax.hist(classifier.y, bins=bins, color="0.1")
    pl = colormaps[cmap]

    bucket_breaks = [0] + [np.searchsorted(bins, i) for i in breaks]
    for c in range(k):
        for b in range(bucket_breaks[c], bucket_breaks[c + 1]):
            patches[b].set_facecolor(pl(c / k))
    if clip is not None:
        histax.set_xlim(*clip)
    histax.set_frame_on(frameon)
    histax.get_yaxis().set_visible(False)
    if tick_params is None:
        tick_params = dict()
    if vlines:
        lim = ax.get_ylim()[1]
        # plot upper limit of each bin
        for i in classifier.bins:
            histax.vlines(i, 0, lim, color=vlinecolor, linewidth=vlinewidth)
    tick_params["labelsize"] = tick_params.get("labelsize", 12)
    histax.tick_params(**tick_params)
    return histax


def _make_location(ax, loc, legend_size=(0.27, 0.2)):
    """
    Construct the location bounds of a legendgram

    Arguments:
    ----------
    ax          :   matplotlib.AxesSubplot
                    axis on which to add a legendgram
    loc         :   string or int
                    valid legend location like that used in matplotlib.pyplot.legend
    legend_size :   tuple or float
                    tuple denoting the length/width of the legendgram in terms
                    of a fraction of the axis. If a float, the legend is assumed
                    square.

    Returns
    -------
    a list [left_anchor, bottom_anchor, width, height] in terms of plot units
    that defines the location and extent of the legendgram.


    """
    position = ax.get_position()
    if isinstance(legend_size, float):
        legend_size = (legend_size, legend_size)
    lw, lh = legend_size
    legend_width = position.width * lw
    legend_height = position.height * lh
    right_offset = position.width - legend_width
    top_offset = position.height - legend_height
    if isinstance(loc, int):
        try:
            loc = inv_lut[loc]
        except KeyError:
            raise KeyError(
                "Legend location {} not recognized. Please choose "
                " from the list of valid matplotlib legend locations."
                "".format(loc)
            )
    if loc.lower() == "lower left" or loc.lower() == "best":
        anchor_x, anchor_y = position.x0, position.y0
    elif loc.lower() == "lower center":
        anchor_x, anchor_y = position.x0 + position.width * 0.5, position.y0
    elif loc.lower() == "lower right":
        anchor_x, anchor_y = position.x0 + right_offset, position.y0
    elif loc.lower() == "center left":
        anchor_x, anchor_y = position.x0, position.y0 + position.height * 0.5
    elif loc.lower() == "center":
        anchor_x, anchor_y = (
            position.x0 + position.width * 0.5,
            position.y0 + position.height * 0.5,
        )
    elif loc.lower() == "center right" or loc.lower() == "right":
        anchor_x, anchor_y = (
            position.x0 + right_offset,
            position.y0 + position.height * 0.5,
        )
    elif loc.lower() == "upper left":
        anchor_x, anchor_y = position.x0, position.y0 + top_offset
    elif loc.lower() == "upper center":
        anchor_x, anchor_y = (
            position.x0 + position.width * 0.5,
            position.y0 + top_offset,
        )
    elif loc.lower() == "upper right":
        anchor_x, anchor_y = position.x0 + right_offset, position.y0 + top_offset
    return [anchor_x, anchor_y, legend_width, legend_height]
