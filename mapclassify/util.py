import numpy as np

from ._classify_API import classify as _classify


def get_rgba(
    values,
    scheme="quantiles",
    cmap="viridis",
    alpha=1,
    nan_color=[255, 255, 255, 255],
    **kwargs,
):
    """Convert array of values into RGBA colors using a colormap and classifier.

    Parameters
    ----------
    values : list-like
        array of input values
    classifier : str, optional
        string description of a mapclassify classifier, by default "quantiles"
    cmap : str, optional
        name of matplotlib colormap to use, by default "viridis"
    alpha : float
        alpha parameter that defines transparency. Should be in the range [0,1]
    nan_color : list, optional
        RGBA color to fill NaN values, by default [255, 255, 255, 255]
    kwargs : dict
        additional keyword arguments are passed to `mapclassify.classify`

    Returns
    -------
    numpy.array
        numpy array (n,4) of lists with each list containing four values that define a
        color using RGBA specification.
    """
    try:
        import pandas as pd
        from matplotlib import colormaps
        from matplotlib.colors import Normalize
    except ImportError as e:
        raise ImportError("This function requires pandas and matplotlib") from e
    if not (alpha <= 1) and (alpha >= 0):
        raise ValueError("alpha must be in the range [0,1]")
    if not pd.api.types.is_list_like(nan_color) and not len(nan_color) == 4:
        raise ValueError("`nan_color` must be list-like of 4 values: (R,G,B,A)")
    if scheme in kwargs:
        kwargs.pop("scheme")

    # only operate on non-NaN values
    v = pd.Series(values, dtype=object)
    legit_indices = v[~v.isna()].index.values
    legit_vals = v.dropna().values
    # transform (non-NaN) values into class bins
    bins = _classify(legit_vals, scheme=scheme, **kwargs).yb

    # create a normalizer using the data's range (not strictly 1-k...)
    norm = Normalize(min(bins), max(bins))
    normalized_vals = norm(bins)

    rgbas = colormaps[cmap](normalized_vals, bytes=True)
    colors = pd.Series(list(rgbas), index=legit_indices).apply(np.array)
    v.update(colors)
    v = v.fillna(f"{nan_color}").apply(np.array)

    return np.stack(v.values)
