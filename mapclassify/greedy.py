"""
greedy - Greedy (topological) coloring for GeoPandas

Copyright (C) 2019 Martin Fleischmann, 2017 Nyall Dawson

"""

import operator

__all__ = [
    "greedy",
]


def _balanced(features, sw, balance="count", min_colors=4):
    """
    Strategy to color features in a way which is visually balanced.

    Algorithm ported from QGIS to be used with GeoDataFrames
    and libpysal weights objects.

    Original algorithm:
    Date                 : February 2017
    Copyright            : (C) 2017 by Nyall Dawson
    Email                : nyall dot dawson at gmail dot com

    Parameters
    ----------
    features : GeoDataFrame
        GeoDataFrame
    sw : libpysal.weights.W
        spatial weights object denoting adjacency of features
    balance : str
        the method of color balancing
    min_colors : int
        the minimal number of colors to be used

    Returns
    -------
    feature_colors : dict
        dictionary with assigned color codes
    """
    feature_colors = {}
    # start with minimum number of colors in pool
    color_pool = set(range(min_colors))

    # calculate count of neighbours
    neighbour_count = sw.cardinalities

    # sort features by neighbour count - handle those with more neighbours first
    sorted_by_count = [
        feature_id
        for feature_id in sorted(
            neighbour_count.items(), key=operator.itemgetter(1), reverse=True
        )
    ]
    # counts for each color already assigned
    color_counts = {}
    color_areas = {}
    for c in color_pool:
        color_counts[c] = 0
        color_areas[c] = 0

    if balance == "centroid":
        features = features.copy()
        features.geometry = features.geometry.centroid
        balance = "distance"

    for (feature_id, n) in sorted_by_count:
        # first work out which already assigned colors are adjacent to this feature
        adjacent_colors = set()
        for neighbour in sw.neighbors[feature_id]:
            if neighbour in feature_colors:
                adjacent_colors.add(feature_colors[neighbour])

        # from the existing colors, work out which are available (ie non-adjacent)
        available_colors = color_pool.difference(adjacent_colors)

        feature_color = -1
        if len(available_colors) == 0:
            # no existing colors available for this feature; add new color and repeat
            min_colors += 1
            return _balanced(features, sw, balance, min_colors)
        else:
            if balance == "count":
                # choose least used available color
                counts = [
                    (c, v) for c, v in color_counts.items() if c in available_colors
                ]
                feature_color = sorted(counts, key=operator.itemgetter(1))[0][0]
                color_counts[feature_color] += 1
            elif balance == "area":
                areas = [
                    (c, v) for c, v in color_areas.items() if c in available_colors
                ]
                feature_color = sorted(areas, key=operator.itemgetter(1))[0][0]
                color_areas[feature_color] += features.loc[feature_id].geometry.area

            elif balance == "distance":
                min_distances = {c: float("inf") for c in available_colors}
                this_feature = features.loc[feature_id].geometry

                # find features for all available colors
                other_features = {
                    f_id: c
                    for (f_id, c) in feature_colors.items()
                    if c in available_colors
                }

                distances = features.loc[other_features.keys()].distance(this_feature)
                # calculate the min distance from this feature to the nearest
                # feature with each assigned color
                for other_feature_id, c in other_features.items():

                    distance = distances.loc[other_feature_id]
                    if distance < min_distances[c]:
                        min_distances[c] = distance

                # choose color such that min distance is maximised!
                # - ie we want MAXIMAL separation between features with the same color
                feature_color = sorted(
                    min_distances, key=min_distances.__getitem__, reverse=True
                )[0]

        feature_colors[feature_id] = feature_color

    return feature_colors


def _geos_sw(features, tolerance=0, silence_warnings=False, resolution=5):
    """
    Generate libpysal spatial weights object based on intersections of features.

    Intersecting features are denoted as neighbours. If tolerance > 0, all features
    within the set tolerance are denoted as neighbours.


    Parameters
    ----------
    features : GeoDataFrame
        GeoDataFrame
    tolerance : float (default 0)
        minimal distance between colors
    silence_warnings : bool (default True)
        silence lilbpysal warnings (if min_distance is set)
    resolution : int (default 5)
        resolution of buffer if tolerance > 0

    Returns
    -------
    W : libpysal.weights.W
        spatial weights object
    """
    try:
        from libpysal.weights import W
    except ImportError:
        raise ImportError("The 'libpysal' package is required.")

    neighbors = {}

    if tolerance > 0:
        features = features.copy()
        features["geometry"] = features.geometry.buffer(tolerance / 2, resolution)

    sindex = features.sindex

    for i, (ix, g) in enumerate(features.geometry.items()):

        possible_matches_index = list(sindex.intersection(g.bounds))
        possible_matches_index.remove(i)
        possible_matches = features.iloc[possible_matches_index]
        precise_matches = possible_matches.loc[possible_matches.intersects(g)]

        neighbors[ix] = list(precise_matches.index)

    return W(neighbors, silence_warnings=silence_warnings)


def greedy(
    gdf,
    strategy="balanced",
    balance="count",
    min_colors=4,
    sw="queen",
    min_distance=None,
    silence_warnings=True,
    interchange=False,
):
    """
    Color GeoDataFrame using various strategies of greedy (topological) colouring.

    Attempts to color a GeoDataFrame using as few colors as possible, where no
    neighbours can have same color as the feature itself. Offers various strategies
    ported from QGIS or implemented within networkX for greedy graph coloring.

    ``greedy`` will return pandas.Series representing assinged color codes.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame
    strategy : str (default 'balanced')
        Determine coloring strategy. Options are ``'balanced'`` for algorithm based on
        QGIS Topological coloring. It is aiming for a visual balance, defined by the
        balance parameter.

        Other options are those supported by networkx.greedy_color:

        * ``'largest_first'``
        * ``'random_sequential'``
        * ``'smallest_last'``
        * ``'independent_set'``
        * ``'connected_sequential_bfs'``
        * ``'connected_sequential_dfs'``
        * ``'connected_sequential'`` (alias for the previous strategy)
        * ``'saturation_largest_first'``
        * ``'DSATUR'`` (alias for the previous strategy)

        For details see
        https://networkx.github.io/documentation/stable/reference/algorithms/
        generated/networkx.algorithms.coloring.greedy_color.html

    balance : str (default 'count')
        If strategy is ``'balanced'``, determine the method of color balancing.

        * ``'count'`` attempts to balance the number of features per each color.
        * ``'area'`` attempts to balance the area covered by each color.
        * ``'centroid'`` attempts to balance the distance between colors based
            on the distance between centroids.
        * ``'distance'`` attempts to balance the distance between colors based
            on the distance between geometries. Slower than ``'centroid'``,
            but more precise.

        ``'centroid'`` and ``'distance'`` are significantly slower than other
        especially for larger GeoDataFrames.

        Apart from ``'count'``, all require CRS to be projected (not in degrees)
        to ensure metric values are correct.

    min_colors: int (default 4)
        If strategy is ``'balanced'``, define the minimal number of colors to be used.

    sw : 'queen', 'rook' or libpysal.weights.W (default 'queen')
        If min_distance is None, one can pass ``'libpysal.weights.W'``
        object denoting neighbors or let greedy to generate one based on
        `'queen'`` or ``'rook'`` contiguity.

    min_distance : float
        Set minimal distance between colors.

        If min_distance is not None, slower algorithm for generating
        spatial weghts is used based on intersection between geometries.
        ``'min_distance'`` is then used as a tolerance of intersection.

    silence_warnings : bool (default True)
        Silence libpysal warnings when creating spatial weights.

    interchange : bool (defaul False)
        Use the color interchange algorithm (applicable for networkx strategies)

        For details see
        https://networkx.github.io/documentation/stable/reference/algorithms/
        generated/networkx.algorithms.coloring.greedy_color.html

    Examples
    --------

    >>> import geopandas as gpd
    >>> from mapclassify import greedy
    >>> world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    >>> africa = world.loc[world.continent == "Africa"].copy()
    >>> africa = africa.to_crs("ESRI:102022").reset_index(drop=True)

    Default:

    >>> africa["greedy_colors"] = greedy(africa)
    >>> africa["greedy_colors"].head()
    0    1
    1    0
    2    0
    3    1
    4    4
    Name: greedy_colors, dtype: int64

    Balanced by area:

    >>> africa["balanced_area"] = greedy(africa, strategy="balanced", balance="area")
    >>> africa["balanced_area"].head()
    0    1
    1    2
    2    0
    3    1
    4    3
    Name: balanced_area, dtype: int64

    Using rook adjacency:

    >>> africa["rook_adjacency"] = greedy(africa, sw="rook")
    >>> africa["rook_adjacency"].tail()
    46    3
    47    0
    48    2
    49    3
    50    1
    Name: rook_adjacency, dtype: int64

    Adding minimal distance between colors:

    >>> africa["min_distance"] = greedy(africa, min_distance=1000000)
    >>> africa["min_distance"].head()
    0    1
    1    8
    2    0
    3    7
    4    4
    Name: min_distance, dtype: int64

    Using different coloring strategy:

    >>> africa["smallest_last"] = greedy(africa, strategy="smallest_last")
    >>> africa["smallest_last"].head()
    0    3
    1    1
    2    1
    3    3
    4    1
    Name: smallest_last, dtype: int64

    Returns
    -------
    color : pd.Series
        pandas.Series representing assinged color codes
    """
    if strategy != "balanced":
        try:
            import networkx as nx

            STRATEGIES = nx.algorithms.coloring.greedy_coloring.STRATEGIES.keys()

        except ImportError:
            raise ImportError("The 'networkx' package is required.")

    try:
        import pandas as pd
    except ImportError:
        raise ImportError("The 'pandas' package is required.")
    try:
        from libpysal.weights import Queen, Rook, W
    except ImportError:
        raise ImportError("The 'libpysal' package is required.")

    if min_distance is not None:
        # TODO: use libpysal's fuzzy_contiguity instead of
        # ``_geos_sw`` once pysal/libpysal#280 is released
        sw = _geos_sw(gdf, tolerance=min_distance, silence_warnings=silence_warnings)

    if not isinstance(sw, W):
        if sw == "queen":
            sw = Queen.from_dataframe(gdf, silence_warnings=silence_warnings)
        elif sw == "rook":
            sw = Rook.from_dataframe(gdf, silence_warnings=silence_warnings)

    if strategy == "balanced":
        color = pd.Series(_balanced(gdf, sw, balance=balance, min_colors=min_colors))

    elif strategy in STRATEGIES:
        color = nx.greedy_color(
            sw.to_networkx(), strategy=strategy, interchange=interchange
        )

    else:
        raise ValueError("{} is not a valid strategy.".format(strategy))

    color = pd.Series(color).sort_index()
    color.index = gdf.index
    return color
