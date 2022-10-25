from geopandas import gpd
from libpysal import examples

import mapclassify
from mapclassify import classify


def _assertions(a, b):
    assert a.k == b.k
    assert a.yb.all() == b.yb.all()
    assert a.bins.all() == b.bins.all()
    assert a.counts.all() == b.counts.all()


def test_classify():
    # data
    link_to_data = examples.get_path("columbus.shp")
    gdf = gpd.read_file(link_to_data)
    x = gdf["HOVAL"].values

    # box_plot
    a = classify(x, "box_plot")
    b = mapclassify.BoxPlot(x)
    _assertions(a, b)

    # EqualInterval
    a = classify(x, "EqualInterval", k=3)
    b = mapclassify.EqualInterval(x, k=3)
    _assertions(a, b)

    # FisherJenks
    a = classify(x, "FisherJenks", k=3)
    b = mapclassify.FisherJenks(x, k=3)
    _assertions(a, b)

    a = classify(x, "FisherJenksSampled", k=3, pct_sampled=0.5, truncate=False)
    b = mapclassify.FisherJenksSampled(x, k=3, pct=0.5, truncate=False)
    _assertions(a, b)

    # headtail_breaks
    a = classify(x, "headtail_breaks")
    b = mapclassify.HeadTailBreaks(x)
    _assertions(a, b)

    # quantiles
    a = classify(x, "quantiles", k=3)
    b = mapclassify.Quantiles(x, k=3)
    _assertions(a, b)

    # percentiles
    a = classify(x, "percentiles", pct=[25, 50, 75, 100])
    b = mapclassify.Percentiles(x, pct=[25, 50, 75, 100])
    _assertions(a, b)

    # JenksCaspall
    a = classify(x, "JenksCaspall", k=3)
    b = mapclassify.JenksCaspall(x, k=3)
    _assertions(a, b)

    a = classify(x, "JenksCaspallForced", k=3)
    b = mapclassify.JenksCaspallForced(x, k=3)
    _assertions(a, b)

    a = classify(x, "JenksCaspallSampled", pct_sampled=0.5)
    b = mapclassify.JenksCaspallSampled(x, pct=0.5)
    _assertions(a, b)

    # natural_breaks, max_p_classifier
    a = classify(x, "natural_breaks")
    b = mapclassify.NaturalBreaks(x)
    _assertions(a, b)

    a = classify(x, "max_p", k=3, initial=50)
    b = mapclassify.MaxP(x, k=3, initial=50)
    _assertions(a, b)

    # std_mean
    a = classify(x, "std_mean", multiples=[-1, -0.5, 0.5, 1])
    b = mapclassify.StdMean(x, multiples=[-1, -0.5, 0.5, 1])
    _assertions(a, b)

    # user_defined
    a = classify(x, "user_defined", bins=[20, max(x)])
    b = mapclassify.UserDefined(x, bins=[20, max(x)])
    _assertions(a, b)
