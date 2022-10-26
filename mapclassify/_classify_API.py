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
):
    """

    Classify your data with `mapclassify.classify`
    Note: Input parameters are dependent on classifier used.

    Parameters
    ----------
    y : array
        (n,1), values to classify
    scheme : str
        pysal.mapclassify classification scheme
    k : int, optional
        The number of classes. Default=5.
    pct  : array, optional
        Percentiles used for classification with `percentiles`.
        Default=[1,10,50,90,99,100]
    pct_sampled : float, optional
        The percentage of n that should form the sample
        (JenksCaspallSampled, FisherJenksSampled)
        If pct is specified such that n*pct > 1000, then pct = 1000./n
    truncate : boolean, optional
        truncate pct_sampled in cases where pct * n > 1000., (Default True)
    hinge : float, optional
        Multiplier for IQR when `BoxPlot` classifier used.
        Default=1.5.
    multiples : array, optional
        The multiples of the standard deviation to add/subtract from
        the sample mean to define the bins using `std_mean`.
        Default=[-2,-1,1,2].
    mindiff : float, optional
        The minimum difference between class breaks
        if using `maximum_breaks` classifier. Deafult =0.
    initial : int
        Number of initial solutions to generate or number of runs
        when using `natural_breaks` or `max_p_classifier`.
        Default =100.
        Note: setting initial to 0 will result in the quickest
        calculation of bins.
    bins : array, optional
        (k,1), upper bounds of classes (have to be monotically
        increasing) if using `user_defined` classifier.
        Default =None, Example =[20, max(y)].

    Returns
    -------
    classifier : pysal.mapclassify.classifier instance
            Object containing bin ids for each observation (.yb),
            upper bounds of each class (.bins), number of classes (.k)
            and number of observations falling in each class (.counts)

    Note: Supported classifiers include: quantiles, box_plot, euqal_interval,
        fisher_jenks, fisher_jenks_sampled, headtail_breaks, jenks_caspall,
        jenks_caspall_sampled, jenks_caspall_forced, max_p, maximum_breaks,
        natural_breaks, percentiles, std_mean, user_defined

    Examples
    --------
    Imports

    >>> from libpysal import examples
    >>> import geopandas as gpd
    >>> from mapclassify import classify

    Load Example Data

    >>> link_to_data = examples.get_path('columbus.shp')
    >>> gdf = gpd.read_file(link_to_data)
    >>> x = gdf['HOVAL'].values

    Classify values by quantiles

    >>> quantiles = classify(x, 'quantiles')

    Classify values by box_plot and set hinge to 2

    >>> box_plot = classify(x, 'box_plot', hinge=2)

    """
    # reformat
    scheme_lower = scheme.lower()
    scheme = scheme_lower.replace("_", "")

    # check if scheme is a valid scheme
    if scheme not in _classifiers:
        raise ValueError(
            "Invalid scheme. Scheme must be in the" " set: %r" % _classifiers.keys()
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
        classifier = _classifiers[scheme](y, multiples)
    elif scheme == "jenkscaspallsampled":
        classifier = _classifiers[scheme](y, k, pct_sampled)
    elif scheme == "maximumbreaks":
        classifier = _classifiers[scheme](y, k, mindiff)

    elif scheme in ["naturalbreaks", "maxp"]:
        classifier = _classifiers[scheme](y, k, initial)
    elif scheme == "userdefined":
        classifier = _classifiers[scheme](y, bins)
    elif scheme in [
        "equalinterval",
        "fisherjenks",
        "jenkscaspall",
        "jenkscaspallforced",
        "quantiles",
    ]:
        classifier = _classifiers[scheme](y, k)

    return classifier
