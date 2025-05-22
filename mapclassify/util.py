import numpy as np

from ._classify_API import classify as _classify


def get_color_array(
    values,
    scheme="quantiles",
    cmap="viridis",
    alpha=1,
    nan_color=[255, 255, 255, 255],
    as_hex=False,
    **kwargs,
):
    """Convert array of values into RGBA or hex colors using a colormap and classifier.
    This function is useful for visualization libraries that require users to provide
    an array of colors for each object (like pydeck or lonboard) but can also be used
    to create a manual column of colors passed to matplotlib.

    Parameters
    ----------
    values : list-like
        array of input values
    scheme : str, optional
        string description of a mapclassify classifier, by default `"quantiles"`
    cmap : str, optional
        name of matplotlib colormap to use, by default "viridis"
    alpha : float
        alpha parameter that defines transparency. Should be in the range [0,1]
    nan_color : list, optional
        RGBA color to fill NaN values, by default [255, 255, 255, 255]
    as_hex: bool, optional
        if True, return a (n,1)-dimensional array of hexcolors instead of a (n,4)
        dimensional array of RGBA values.
    kwargs : dict
        additional keyword arguments are passed to `mapclassify.classify`

    Returns
    -------
    numpy.array
        numpy array (aligned with the input array) defining a color for each row. If
        `as_hex` is False, the array is :math:`(n,4)` holding an array of RGBA values in
        each row. If `as_hex` is True, the array is :math:`(n,1)` holding a hexcolor in
        each row.

    """
    try:
        import pandas as pd
        from matplotlib import colormaps
        from matplotlib.colors import Normalize, to_hex
    except ImportError as e:
        raise ImportError("This function requires pandas and matplotlib") from e
    if not (alpha <= 1) and (alpha >= 0):
        raise ValueError("alpha must be in the range [0,1]")
    if not pd.api.types.is_list_like(nan_color) and not len(nan_color) == 4:
        raise ValueError("`nan_color` must be list-like of 4 values: (R,G,B,A)")

    # only operate on non-NaN values
    v = pd.Series(values, dtype=object)
    legit_indices = v[~v.isna()].index.values
    legit_vals = v.dropna().values
    bogus_indices = v[v.isna()].index.values  # stash these for use later
    # transform (non-NaN) values into class bins
    bins = _classify(legit_vals, scheme=scheme, **kwargs).yb

    # normalize using the data's range (not strictly 1-k if classifier is degenerate)
    norm = Normalize(min(bins), max(bins))
    normalized_vals = norm(bins)

    # generate RBGA array and convert to series
    rgbas = colormaps[cmap](normalized_vals, bytes=True, alpha=alpha)
    colors = pd.Series(list(rgbas), index=legit_indices).apply(np.array)
    nan_colors = pd.Series(
        [nan_color for i in range(len(bogus_indices))], index=bogus_indices
    ).apply(lambda x: np.array(x).astype(np.uint8))

    # put colors in their correct places and fill empty with specified color
    v.update(colors)
    v.update(nan_colors)

    # convert to hexcolors if preferred
    if as_hex:
        colors = v.apply(lambda x: to_hex(x / 255.0))
        return colors.values
    return np.stack(v.values)
