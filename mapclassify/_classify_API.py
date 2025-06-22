from .classifiers import (
    BoxPlot,
    EqualInterval,
    FisherJenks,
    FisherJenksSampled,
    HeadTailBreaks,
    JenksCaspall,
    JenksCaspallForced,
    JenksCaspallSampled,
    MaximumBreaks,
    MaxP,
    NaturalBreaks,
    Percentiles,
    PrettyBreaks,
    Quantiles,
    StdMean,
    UserDefined,
)

__author__ = "Stefanie Lumnitz <stefanie.lumitz@gmail.com>"


_classifiers = {
    "boxplot": BoxPlot,
    "equalinterval": EqualInterval,
    "fisherjenks": FisherJenks,
    "fisherjenkssampled": FisherJenksSampled,
    "headtailbreaks": HeadTailBreaks,
    "jenkscaspall": JenksCaspall,
    "jenkscaspallforced": JenksCaspallForced,
    "jenkscaspallsampled": JenksCaspallSampled,
    "maxp": MaxP,
    "maximumbreaks": MaximumBreaks,
    "naturalbreaks": NaturalBreaks,
    "quantiles": Quantiles,
    "percentiles": Percentiles,
    "prettybreaks": PrettyBreaks,
    "stdmean": StdMean,
    "userdefined": UserDefined,
}


def classify(
    y,
    scheme,
    k=5,
    pct=[1, 10, 50, 90, 99, 100],
    pct_sampled=0.10,
    truncate=True,
    hinge=1.5,
    multiples=[-2, -1, 1, 2],
    mindiff=0,
    initial=100,
    bins=None,
    lowest=None,
    anchor=False,
):
    """

    Classify your data with ``mapclassify.classify``.
    Input parameters are dependent on classifier used.

    Parameters
    ----------

    y : numpy.array
        :math:`(n,1)`, values to classify.
    scheme : str
        ``pysal.mapclassify`` classification scheme.
    k : int (default 5)
        The number of classes.
    pct  : numpy.array (default [1, 10, 50, 90, 99, 100])
        Percentiles used for classification with ``percentiles``.
    pct_sampled : float default (0.10)
        The percentage of n that should form the sample
        (``JenksCaspallSampled``, ``FisherJenksSampled``)
        If ``pct`` is specified such that ``n*pct > 1000``, then ``pct=1000``.
    truncate : bool (default True)
        Truncate ``pct_sampled`` in cases where ``pct * n > 1000``.
    hinge : float (default 1.5)
        Multiplier for *IQR* when ``BoxPlot`` classifier used.
    multiples : numpy.array (default [-2,-1,1,2])
        The multiples of the standard deviation to add/subtract from
        the sample mean to define the bins using ``std_mean``.
    mindiff : float (default is 0)
        The minimum difference between class breaks
        if using ``maximum_breaks`` classifier.
    initial : int (default 100)
        Number of initial solutions to generate or number of runs when using
        ``natural_breaks`` or ``max_p_classifier``. Setting initial to ``0``
        will result in the quickest calculation of bins.
    bins : numpy.array (default None)
        :math:`(k,1)`, upper bounds of classes (have to be monotically
        increasing) if using ``user_defined`` classifier.
        Default is ``None``. For example: ``[20, max(y)]``.
    lowest : float (default None)
        Scalar minimum value of lowest class. Default is to set the minimum
        to ``-inf`` if  ``y.min()`` > first upper bound (which will override
        the default), otherwise minimum is set to ``y.min()``.
    anchor : bool (default False)
        Anchor upper bound of one class to the sample mean.

    Returns
    -------
    classifier : mapclassify.classifiers.MapClassifier
        Object containing bin ids for each observation (``.yb``),
        upper bounds of each class (``.bins``), number of classes (``.k``)
        and number of observations falling in each class (``.counts``).

    Notes
    -----

    Supported classifiers include:

    * ``quantiles``
    * ``boxplot``
    * ``equalinterval``
    * ``fisherjenks``
    * ``fisherjenkssampled``
    * ``headtailbreaks``
    * ``jenkscaspall``
    * ``jenkscaspallsampled``
    * ``jenks_caspallforced``
    * ``maxp``
    * ``maximumbreaks``
    * ``naturalbreaks``
    * ``percentiles``
    * ``prettybreaks``
    * ``stdmean``
    * ``userdefined``

    Examples
    --------

    >>> import libpysal
    >>> import geopandas
    >>> from mapclassify import classify

    Load example data.

    >>> link_to_data = libpysal.examples.get_path("columbus.shp")
    >>> gdf = geopandas.read_file(link_to_data)
    >>> x = gdf['HOVAL'].values

    Classify values by quantiles.

    >>> quantiles = classify(x, "quantiles")

    Classify values by box_plot and set hinge to ``2``.

    >>> box_plot = classify(x, 'box_plot', hinge=2)
    >>> box_plot
    BoxPlot
    <BLANKLINE>
       Interval      Count
    ----------------------
    ( -inf, -9.50] |     0
    (-9.50, 25.70] |    13
    (25.70, 33.50] |    12
    (33.50, 43.30] |    12
    (43.30, 78.50] |     9
    (78.50, 96.40] |     3

    """

    # reformat
    scheme_lower = scheme.lower()
    scheme = scheme_lower.replace("_", "")

    # check if scheme is a valid scheme
    if scheme not in _classifiers:
        raise ValueError(
            f"Invalid scheme: '{scheme}'\n"
            f"Scheme must be in the set: {_classifiers.keys()}"
        )

    elif scheme == "boxplot":
        classifier = _classifiers[scheme](y, hinge)

    elif scheme == "fisherjenkssampled":
        classifier = _classifiers[scheme](y, k, pct_sampled, truncate)

    elif scheme == "headtailbreaks":
        classifier = _classifiers[scheme](y)

    elif scheme == "percentiles":
        classifier = _classifiers[scheme](y, pct)

    elif scheme == "stdmean":
        classifier = _classifiers[scheme](y, multiples, anchor)

    elif scheme == "jenkscaspallsampled":
        classifier = _classifiers[scheme](y, k, pct_sampled)

    elif scheme == "maximumbreaks":
        classifier = _classifiers[scheme](y, k, mindiff)

    elif scheme in ["naturalbreaks", "maxp"]:
        classifier = _classifiers[scheme](y, k, initial)

    elif scheme == "userdefined":
        classifier = _classifiers[scheme](y, bins, lowest)

    elif scheme in [
        "equalinterval",
        "fisherjenks",
        "jenkscaspall",
        "jenkscaspallforced",
        "quantiles",
        "prettybreaks",
    ]:
        classifier = _classifiers[scheme](y, k)

    return classifier
