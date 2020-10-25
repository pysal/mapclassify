import mapclassify

__author__ = ("Stefanie Lumnitz <stefanie.lumitz@gmail.com>")

_classifiers = {
    'box_plot': mapclassify.BoxPlot,
    'equal_interval': mapclassify.EqualInterval,
    'fisher_jenks': mapclassify.FisherJenks,
    'headtail_breaks': mapclassify.HeadTailBreaks,
    'jenks_caspall': mapclassify.JenksCaspall,
    'jenks_caspall_forced': mapclassify.JenksCaspallForced,
    'max_p_classifier': mapclassify.MaxP,
    'maximum_breaks': mapclassify.MaximumBreaks,
    'natural_breaks': mapclassify.NaturalBreaks,
    'quantiles': mapclassify.Quantiles,
    'percentiles': mapclassify.Percentiles,
    'std_mean': mapclassify.StdMean,
    'user_defined': mapclassify.UserDefined,
    }

def classify(y, scheme, k=5, pct=[1,10,50,90,99,100],
                    hinge=1.5, multiples=[-2,-1,1,2], mindiff=0,
                    initial=100, bins=None):
    """
    Classify your data with `pysal.mapclassify.classify`
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
    bins : pysal.mapclassify instance
        Object containing bin ids for each observation (.yb),
        upper bounds of each class (.bins), number of classes (.k)
        and number of onservations falling in each class (.counts)
    
    Note: Supported classifiers include: quantiles, box_plot, euqal_interval,
        fisher_jenks, headtail_breaks, jenks_caspall, jenks_caspall_forced,
        max_p_classifier, maximum_breaks, natural_breaks, percentiles, std_mean,
        user_defined
    
    Examples
    --------
    Imports
    
    >>> from libpysal import examples
    >>> import geopandas as gpd
    >>> from splot.mapping import mapclassify_bin
    
    Load Example Data
    
    >>> link_to_data = examples.get_path('columbus.shp')
    >>> gdf = gpd.read_file(link_to_data)
    >>> x = gdf['HOVAL'].values
    
    Classify values by quantiles
    
    >>> quantiles = mapclassify_bin(x, 'quantiles')
    
    Classify values by box_plot and set hinge to 2
    
    >>> box_plot = mapclassify_bin(x, 'box_plot', hinge=2)
    
    """
    classifier = classifier.lower()
    if scheme not in _classifiers:
        raise ValueError("Invalid scheme. Scheme must be in the"
                         " set: %r" % _classifiers.keys())
    elif scheme == 'box_plot':
        bins = _classifiers[classifier](y, hinge)
    elif scheme == 'headtail_breaks':
        bins = _classifiers[classifier](y)
    elif scheme == 'percentiles':
        bins = _classifiers[classifier](y, pct)
    elif scheme == 'std_mean':
        bins = _classifiers[classifier](y, multiples)
    elif scheme == 'maximum_breaks':
        bins = _classifiers[classifier](y, k, mindiff)
    elif scheme in ['natural_breaks', 'max_p_classifier']:
        bins = _classifiers[classifier](y, k, initial)
    elif scheme == 'user_defined':
        bins = _classifiers[classifier](y, bins)
    else:
        bins = _classifiers[classifier](y, k)
    return bins