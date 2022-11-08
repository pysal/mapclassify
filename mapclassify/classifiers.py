"""
A module of classification schemes for choropleth mapping.
"""
import copy
import functools
from warnings import warn as Warn

import numpy as np
import scipy.stats as stats
from sklearn.cluster import KMeans as KMEANS

__author__ = "Sergio J. Rey"

__all__ = [
    "MapClassifier",
    "quantile",
    "BoxPlot",
    "EqualInterval",
    "FisherJenks",
    "FisherJenksSampled",
    "JenksCaspall",
    "JenksCaspallForced",
    "JenksCaspallSampled",
    "HeadTailBreaks",
    "MaxP",
    "MaximumBreaks",
    "NaturalBreaks",
    "Quantiles",
    "Percentiles",
    "StdMean",
    "UserDefined",
    "gadf",
    "KClassifiers",
    "CLASSIFIERS",
]

CLASSIFIERS = (
    "BoxPlot",
    "EqualInterval",
    "FisherJenks",
    "FisherJenksSampled",
    "HeadTailBreaks",
    "JenksCaspall",
    "JenksCaspallForced",
    "JenksCaspallSampled",
    "MaxP",
    "MaximumBreaks",
    "NaturalBreaks",
    "Quantiles",
    "Percentiles",
    "StdMean",
    "UserDefined",
)

K = 5  # default number of classes in any map scheme with this as an argument
SEEDRANGE = 1000000  # range for drawing random ints from for Natural Breaks


FMT = "{:.2f}"

try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(type, cache):
        def decorator_njit(func):
            @functools.wraps(func)
            def wrapper_decorator(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper_decorator

        return decorator_njit


def _format_intervals(mc, fmt="{:.0f}"):
    """
    Helper methods to format legend intervals


    Parameters
    ----------

    mc: MapClassifier

    fmt: str
         specification of formatting for legend

    Returns
    -------
    tuple:
         edges: list
                k strings for class intervals
         max_width: int
                length of largest interval string
         lower_open: boolean
                True: lower bound of first interval is open
                False: lower bound of first interval is closed

    Notes
    -----

    For some classifiers, it is possible that the upper bound of the first
    interval is less than the minimum value of the attribute that is being
    classified. In these cases ``lower_open=True`` and the lower bound of the
    interval is set to ``np.NINF```.

    """

    lowest = mc.y.min()
    if hasattr(mc, "lowest"):
        if mc.lowest is not None:
            lowest = mc.lowest
    lower_open = False
    if lowest > mc.bins[0]:
        lowest = np.NINF
        lower_open = True
    edges = [lowest]
    edges.extend(mc.bins)
    edges = [fmt.format(edge) for edge in edges]
    max_width = max([len(edge) for edge in edges])
    return edges, max_width, lower_open


def _get_mpl_labels(mc, fmt="{:.1f}"):
    """
    Helper method to format legend intervals for matplotlib (and geopandas)

    Parameters
    ----------

    mc: MapClassifier

    fmt: str
         specification of formatting for legend

    Returns
    -------
    intervals: list
             k strings for class intervals
    """
    edges, max_width, lower_open = _format_intervals(mc, fmt)
    k = len(edges) - 1
    left = ["["]
    if lower_open:
        left = ["("]
    left.extend("(" * k)
    right = "]" * (k + 1)
    lower = ["{:>{width}}".format(edges[i], width=max_width) for i in range(k)]
    upper = ["{:>{width}}".format(edges[i], width=max_width) for i in range(1, k + 1)]
    lower = [_l + r for _l, r in zip(left, lower)]
    upper = [_l + r for _l, r in zip(upper, right)]
    intervals = [_l + ", " + r for _l, r in zip(lower, upper)]
    return intervals


def _get_table(mc, fmt="{:.2f}"):
    """
    Helper function to generate tabular classification report

    Parameters
    ----------

    mc: MapClassifier

    fmt: str
         specification of formatting for legend

    Returns
    -------
    table: string
           formatted table of classification results

    """
    intervals = _get_mpl_labels(mc, fmt)
    interval_width = len(intervals[0])
    counts = list(map(str, mc.counts))
    count_width = max([len(count) for count in counts])
    count_width = max(count_width, len("count"))
    interval_width = max(interval_width, len("interval"))
    header = f"{'Interval' : ^{interval_width}}"
    header += " " * 3 + f"{'Count' : >{count_width}}"
    title = mc.name
    header += "\n" + "-" * len(header)
    table = [title, "", header]
    for i, interval in enumerate(intervals):
        row = f"{interval} | {counts[i] : >{count_width}}"
        table.append(row)
    return "\n".join(table)


def head_tail_breaks(values, cuts):
    """
    head tail breaks helper function
    """
    values = np.array(values)
    mean = np.mean(values)
    if len(cuts) > 0 and cuts[-1] == mean:  # fix floating point issue #117
        return cuts
    cuts.append(mean)
    if len(set(values)) > 1:
        return head_tail_breaks(values[values > mean], cuts)
    return cuts


def quantile(y, k=4):
    """
    Calculates the quantiles for an array

    Parameters
    ----------
    y : array
        (n,1), values to classify
    k : int
        number of quantiles

    Returns
    -------
    q         : array
                (n,1), quantile values

    Examples
    --------
    >>> import numpy as np
    >>> import mapclassify as mc
    >>> x = np.arange(1000)
    >>> mc.classifiers.quantile(x)
    array([249.75, 499.5 , 749.25, 999.  ])
    >>> mc.classifiers.quantile(x, k=3)
    array([333., 666., 999.])

    Note that if there are enough ties that the quantile values repeat, we
    collapse to pseudo quantiles in which case the number of classes will be
    less than k

    >>> x = [1.0] * 100
    >>> x.extend([3.0] * 40)
    >>> len(x)
    140
    >>> y = np.array(x)
    >>> mc.classifiers.quantile(y)
    array([1., 3.])
    """

    w = 100.0 / k
    p = np.arange(w, 100 + w, w)
    if p[-1] > 100.0:
        p[-1] = 100.0
    q = np.array([stats.scoreatpercentile(y, pct) for pct in p])
    q = np.unique(q)
    k_q = len(q)
    if k_q < k:
        Warn(
            "Warning: Not enough unique values in array to form k classes", UserWarning
        )
        Warn("Warning: setting k to %d" % k_q, UserWarning)
    return q


def binC(y, bins):
    """
    Bin categorical/qualitative data

    Parameters
    ----------
    y    : array
           (n,q), categorical values
    bins : array
           (k,1),  unique values associated with each bin

    Return
    ------
    b : array
        (n,q), bin membership, values between 0 and k-1

    Examples
    --------
    >>> import numpy as np
    >>> import mapclassify as mc
    >>> np.random.seed(1)
    >>> x = np.random.randint(2, 8, (10, 3))
    >>> bins = list(range(2, 8))
    >>> x
    array([[7, 5, 6],
           [2, 3, 5],
           [7, 2, 2],
           [3, 6, 7],
           [6, 3, 4],
           [6, 7, 4],
           [6, 5, 6],
           [4, 6, 7],
           [4, 6, 3],
           [3, 2, 7]])
    >>> y = mc.classifiers.binC(x, bins)
    >>> y
    array([[5, 3, 4],
           [0, 1, 3],
           [5, 0, 0],
           [1, 4, 5],
           [4, 1, 2],
           [4, 5, 2],
           [4, 3, 4],
           [2, 4, 5],
           [2, 4, 1],
           [1, 0, 5]])
    """

    if np.ndim(y) == 1:
        k = 1
        n = np.shape(y)[0]
    else:
        n, k = np.shape(y)
    b = np.zeros((n, k), dtype="int")
    for i, bin in enumerate(bins):
        b[np.nonzero(y == bin)] = i

    # check for non-binned items and warn if needed
    vals = set(y.flatten())
    for val in vals:
        if val not in bins:
            Warn("value not in bin: {}".format(val), UserWarning)
            Warn("bins: {}".format(bins), UserWarning)

    return b


def bin(y, bins):
    """
    bin interval/ratio data

    Parameters
    ----------
    y : array
        (n,q), values to bin
    bins : array
           (k,1), upper bounds of each bin (monotonic)

    Returns
    -------
    b : array
        (n,q), values of values between 0 and k-1

    Examples
    --------
    >>> import numpy as np
    >>> import mapclassify as mc
    >>> np.random.seed(1)
    >>> x = np.random.randint(2, 20, (10, 3))
    >>> bins = [10, 15, 20]
    >>> b = mc.classifiers.bin(x, bins)
    >>> x
    array([[ 7, 13, 14],
           [10, 11, 13],
           [ 7, 17,  2],
           [18,  3, 14],
           [ 9, 15,  8],
           [ 7, 13, 12],
           [16,  6, 11],
           [19,  2, 15],
           [11, 11,  9],
           [ 3,  2, 19]])
    >>> b
    array([[0, 1, 1],
           [0, 1, 1],
           [0, 2, 0],
           [2, 0, 1],
           [0, 1, 0],
           [0, 1, 1],
           [2, 0, 1],
           [2, 0, 1],
           [1, 1, 0],
           [0, 0, 2]])
    """
    if np.ndim(y) == 1:
        k = 1
        n = np.shape(y)[0]
    else:
        n, k = np.shape(y)
    b = np.zeros((n, k), dtype="int")
    i = len(bins)
    if type(bins) != list:
        bins = bins.tolist()
    binsc = copy.copy(bins)
    while binsc:
        i -= 1
        c = binsc.pop(-1)
        b[np.nonzero(y <= c)] = i
    return b


def bin1d(x, bins):
    """
    Place values of a 1-d array into bins and determine counts of values in
    each bin

    Parameters
    ----------
    x : array
        (n, 1), values to bin
    bins : array
           (k,1), upper bounds of each bin (monotonic)

    Returns
    -------
    binIds : array
             1-d array of integer bin Ids

    counts : int
            number of elements of x falling in each bin

    Examples
    --------
    >>> import numpy as np
    >>> import mapclassify as mc
    >>> x = np.arange(100, dtype = 'float')
    >>> bins = [25, 74, 100]
    >>> binIds, counts = mc.classifiers.bin1d(x, bins)
    >>> binIds
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    >>> list(counts)
    [26, 49, 25]
    """
    left = [-float("inf")]
    left.extend(bins[0:-1])
    right = bins
    cuts = list(zip(left, right))
    k = len(bins)
    binIds = np.zeros(x.shape, dtype="int")
    while cuts:
        k -= 1
        l, r = cuts.pop(-1)
        binIds += (x > l) * (x <= r) * k
    counts = np.bincount(binIds, minlength=len(bins))
    return (binIds, counts)


def load_example():
    """
    Helper function for doc tests
    """
    from .datasets import calemp

    return calemp.load()


def _kmeans(y, k=5, n_init=10):
    """
    Helper function to do k-means in one dimension

    Parameters
    ----------

    y       : array
              (n,1), values to classify
    k       : int
              number of classes to form

    n_init : int, default: 10
              number of initial  solutions. Best of initial results is returned.
    """

    y = y * 1.0  # KMEANS needs float or double dtype
    y.shape = (-1, 1)
    result = KMEANS(n_clusters=k, init="k-means++", n_init=n_init).fit(y)
    class_ids = result.labels_
    centroids = result.cluster_centers_
    binning = []
    for c in range(k):
        values = y[class_ids == c]
        binning.append([values.max(), len(values)])
    binning = np.array(binning)
    binning = binning[binning[:, 0].argsort()]
    cuts = binning[:, 0]

    y_cent = np.zeros_like(y)
    for c in range(k):
        y_cent[class_ids == c] = centroids[c]
    diffs = y - y_cent
    diffs *= diffs

    return class_ids, cuts, diffs.sum(), centroids


def natural_breaks(values, k=5, init=10):
    """
    natural breaks helper function

    Jenks natural breaks is kmeans in one dimension

    Parameters
    ----------

    values : array
             (n, 1) values to bin

    k : int
        Number of classes

    init: int, default:10
        Number of different solutions to obtain using different centroids.
        Best solution is returned.


    """
    values = np.array(values)
    uv = np.unique(values)
    uvk = len(uv)
    if uvk < k:
        Warn(
            "Warning: Not enough unique values in array to form k classes", UserWarning
        )
        Warn("Warning: setting k to %d" % uvk, UserWarning)
        k = uvk
    kres = _kmeans(values, k, n_init=init)
    sids = kres[-1]  # centroids
    fit = kres[-2]
    class_ids = kres[0]
    cuts = kres[1]
    return (sids, class_ids, fit, cuts)


@njit("f8[:](f8[:], u2)", cache=True)
def _fisher_jenks_means(values, classes=5):
    """
    Jenks Optimal (Natural Breaks) algorithm implemented in Python.

    Notes
    -----
    The original Python code comes from here:
    http://danieljlewis.org/2010/06/07/jenks-natural-breaks-algorithm-in-python/
    and is based on a JAVA and Fortran code available here:
    https://stat.ethz.ch/pipermail/r-sig-geo/2006-March/000811.html

    Returns class breaks such that classes are internally homogeneous while
    assuring heterogeneity among classes.

    """
    n_data = len(values)
    mat1 = np.zeros((n_data + 1, classes + 1), dtype=np.int32)
    mat2 = np.zeros((n_data + 1, classes + 1), dtype=np.float32)
    mat1[1, 1:] = 1
    mat2[2:, 1:] = np.inf

    v = np.float32(0)
    for _l in range(2, len(values) + 1):
        s1 = np.float32(0)
        s2 = np.float32(0)
        w = np.float32(0)
        for m in range(1, _l + 1):
            i3 = _l - m + 1
            val = np.float32(values[i3 - 1])
            s2 += val * val
            s1 += val
            w += np.float32(1)
            v = s2 - (s1 * s1) / w
            i4 = i3 - 1
            if i4 != 0:
                for j in range(2, classes + 1):
                    if mat2[_l, j] >= (v + mat2[i4, j - 1]):
                        mat1[_l, j] = i3
                        mat2[_l, j] = v + mat2[i4, j - 1]
        mat1[_l, 1] = 1
        mat2[_l, 1] = v

    k = len(values)

    kclass = np.zeros(classes + 1, dtype=values.dtype)
    kclass[classes] = values[len(values) - 1]
    kclass[0] = values[0]
    for countNum in range(classes, 1, -1):
        pivot = mat1[k, countNum]
        id = int(pivot - 2)
        kclass[countNum - 1] = values[id]
        k = int(pivot - 1)
    return np.delete(kclass, 0)


class MapClassifier(object):
    r"""
    Abstract class for all map classifications :cite:`Slocum_2009`

    For an array :math:`y` of :math:`n` values, a map classifier places each
    value :math:`y_i` into one of :math:`k` mutually exclusive and exhaustive
    classes.  Each classifer defines the classes based on different criteria,
    but in all cases the following hold for the classifiers in PySAL:

    .. math:: C_j^l < y_i \le C_j^u \  \forall  i \in C_j

    where :math:`C_j` denotes class :math:`j` which has lower bound
          :math:`C_j^l` and upper bound :math:`C_j^u`.

    Map Classifiers Supported

    * :class:`mapclassify.classifiers.BoxPlot`
    * :class:`mapclassify.classifiers.EqualInterval`
    * :class:`mapclassify.classifiers.FisherJenks`
    * :class:`mapclassify.classifiers.FisherJenksSampled`
    * :class:`mapclassify.classifiers.HeadTailBreaks`
    * :class:`mapclassify.classifiers.JenksCaspall`
    * :class:`mapclassify.classifiers.JenksCaspallForced`
    * :class:`mapclassify.classifiers.JenksCaspallSampled`
    * :class:`mapclassify.classifiers.MaxP`
    * :class:`mapclassify.classifiers.MaximumBreaks`
    * :class:`mapclassify.classifiers.NaturalBreaks`
    * :class:`mapclassify.classifiers.Quantiles`
    * :class:`mapclassify.classifiers.Percentiles`
    * :class:`mapclassify.classifiers.StdMean`
    * :class:`mapclassify.classifiers.UserDefined`

    Utilities:

    In addition to the classifiers, there are several utility functions that
    can be used to evaluate the properties of a specific classifier,
    or for automatic selection of a classifier and
    number of classes.

    * :func:`mapclassify.classifiers.gadf`
    * :class:`mapclassify.classifiers.K_classifiers`

    """

    def __init__(self, y):
        y = np.asarray(y).flatten()
        self.name = "Map Classifier"
        self.fmt = FMT
        self.y = y
        self._classify()
        self._summary()

    def get_fmt(self):
        return self._fmt

    def set_fmt(self, fmt):
        self._fmt = fmt

    fmt = property(get_fmt, set_fmt)

    def _summary(self):
        yb = self.yb
        self.classes = [np.nonzero(yb == c)[0].tolist() for c in range(self.k)]
        self.tss = self.get_tss()
        self.adcm = self.get_adcm()
        self.gadf = self.get_gadf()

    def _classify(self):
        self._set_bins()
        self.yb, self.counts = bin1d(self.y, self.bins)

    def _update(self, data, *args, **kwargs):
        """
        The only thing that *should* happen in this function is
        1. input sanitization for pandas
        2. classification/reclassification.

        Using their __init__ methods, all classifiers can re-classify given
        different input parameters or additional data.

        If you've got a cleverer updating equation than the intial estimation
        equation, remove the call to self.__init__ below and replace it with
        the updating function.
        """
        if data is not None:
            data = np.asarray(data).flatten()
            data = np.append(data.flatten(), self.y)
        else:
            data = self.y
        self.__init__(data, *args, **kwargs)

    @classmethod
    def make(cls, *args, **kwargs):
        """
        Configure and create a classifier that will consume data and produce
        classifications, given the configuration options specified by this
        function.

        Note that this like a *partial application* of the relevant class
        constructor. `make` creates a function that returns classifications; it
        does not actually do the classification.

        If you want to classify data directly, use the appropriate class
        constructor, like Quantiles, Max_Breaks, etc.

        If you *have* a classifier object, but want to find which bins new data
        falls into, use find_bin.

        Parameters
        ----------
        *args           : required positional arguments
                          all positional arguments required by the classifier,
                          excluding the input data.
        rolling         : bool
                          a boolean configuring the outputted classifier to use
                          a rolling classifier rather than a new classifier for
                          each input. If rolling, this adds the current data to
                          all of the previous data in the classifier, and
                          rebalances the bins, like a running median
                          computation.
        return_object   : bool
                          a boolean configuring the outputted classifier to
                          return the classifier object or not
        return_bins     : bool
                          a boolean configuring the outputted classifier to
                          return the bins/breaks or not
        return_counts   : bool
                          a boolean configuring the outputted classifier to
                          return the histogram of objects falling into each bin
                          or not

        Returns
        -------
        A function that consumes data and returns their bins (and object,
        bins/breaks, or counts, if requested).

        Note
        ----
        This is most useful when you want to run a classifier many times
        with a given configuration, such as when classifying many columns of an
        array or dataframe using the same configuration.

        Examples
        --------
        >>> import libpysal as ps
        >>> import mapclassify as mc
        >>> import geopandas as gpd
        >>> df = gpd.read_file(ps.examples.get_path('columbus.dbf'))
        >>> classifier = mc.Quantiles.make(k=9)
        >>> cl = df[['HOVAL', 'CRIME', 'INC']].apply(classifier)
        >>> cl["HOVAL"].values[:10]
        array([8, 7, 2, 4, 1, 3, 8, 5, 7, 8])
        >>> cl["CRIME"].values[:10]
        array([0, 1, 3, 4, 6, 2, 0, 5, 3, 4])
        >>> cl["INC"].values[:10]
        array([7, 8, 5, 0, 3, 5, 0, 3, 6, 4])
        >>> import pandas as pd; from numpy import linspace as lsp
        >>> data = [lsp(3,8,num=10), lsp(10, 0, num=10), lsp(-5, 15, num=10)]
        >>> data = pd.DataFrame(data).T
        >>> data
                  0          1          2
        0  3.000000  10.000000  -5.000000
        1  3.555556   8.888889  -2.777778
        2  4.111111   7.777778  -0.555556
        3  4.666667   6.666667   1.666667
        4  5.222222   5.555556   3.888889
        5  5.777778   4.444444   6.111111
        6  6.333333   3.333333   8.333333
        7  6.888889   2.222222  10.555556
        8  7.444444   1.111111  12.777778
        9  8.000000   0.000000  15.000000
        >>> data.apply(mc.Quantiles.make(rolling=True))
           0  1  2
        0  0  4  0
        1  0  4  0
        2  1  4  0
        3  1  3  0
        4  2  2  1
        5  2  1  2
        6  3  1  4
        7  3  0  4
        8  4  0  4
        9  4  0  4
        >>> dbf = ps.io.open(ps.examples.get_path('baltim.dbf'))
        >>> data = dbf.by_col_array('PRICE', 'LOTSZ', 'SQFT')
        >>> my_bins = [1, 10, 20, 40, 80]
        >>> cl = [mc.UserDefined.make(bins=my_bins)(a) for a in data.T]
        >>> len(cl)
        3
        >>> cl[0][:10]
        array([4, 5, 5, 5, 4, 4, 5, 4, 4, 5])
        """

        # only flag overrides return flag
        to_annotate = copy.deepcopy(kwargs)
        return_object = kwargs.pop("return_object", False)
        return_bins = kwargs.pop("return_bins", False)
        return_counts = kwargs.pop("return_counts", False)

        rolling = kwargs.pop("rolling", False)
        if rolling:
            #  just initialize a fake classifier
            data = list(range(10))
            cls_instance = cls(data, *args, **kwargs)
            #  and empty it, since we'll be using the update
            cls_instance.y = np.array([])
        else:
            cls_instance = None

        #  wrap init in a closure to make a consumer.
        #  Qc Na: "Objects/Closures are poor man's Closures/Objects"
        def classifier(data, cls_instance=cls_instance):
            if rolling:
                cls_instance.update(data, inplace=True, **kwargs)
                yb = cls_instance.find_bin(data)
            else:
                cls_instance = cls(data, *args, **kwargs)
                yb = cls_instance.yb
            outs = [yb, None, None, None]
            outs[1] = cls_instance if return_object else None
            outs[2] = cls_instance.bins if return_bins else None
            outs[3] = cls_instance.counts if return_counts else None
            outs = [a for a in outs if a is not None]
            if len(outs) == 1:
                return outs[0]
            else:
                return outs

        #  for debugging/jic, keep around the kwargs.
        #  in future, we might want to make this a thin class, so that we can
        #  set a custom repr. Call the class `Binner` or something, that's a
        #  pre-configured Classifier that just consumes data, bins it, &
        #  possibly updates the bins.
        classifier._options = to_annotate
        return classifier

    def update(self, y=None, inplace=False, **kwargs):
        """
        Add data or change classification parameters.

        Parameters
        ----------
        y       :   array
                    (n,1) array of data to classify
        inplace :   bool
                    whether to conduct the update in place or to return a copy
                    estimated from the additional specifications.

        Additional parameters provided in **kwargs are passed to the init
        function of the class. For documentation, check the class constructor.
        """
        kwargs.update({"k": kwargs.pop("k", self.k)})
        if inplace:
            self._update(y, **kwargs)
        else:
            new = copy.deepcopy(self)
            new._update(y, **kwargs)
            return new

    def __str__(self):
        return self.table()

    def __repr__(self):
        return self.table()

    def table(self):
        fmt = self.fmt
        return _get_table(self, fmt=fmt)

    def __call__(self, *args, **kwargs):
        """
        This will allow the classifier to be called like it's a function.

        Whether or not we want to make this be "find_bin" or "update" is a
        design decision.

        I like this as find_bin, since a classifier's job should be to classify
        the data given to it using the rules estimated from the `_classify()`
        function.
        """
        return self.find_bin(*args)

    def get_tss(self):
        """
        Total sum of squares around class means

        Returns sum of squares over all class means
        """
        tss = 0
        for class_def in self.classes:
            if len(class_def) > 0:
                yc = self.y[class_def]
                css = yc - yc.mean()
                css *= css
                tss += sum(css)
        return tss

    def _set_bins(self):
        pass

    def get_adcm(self):
        """
        Absolute deviation around class median (ADCM).

        Calculates the absolute deviations of each observation about its class
        median as a measure of fit for the classification method.

        Returns sum of ADCM over all classes
        """
        adcm = 0
        for class_def in self.classes:
            if len(class_def) > 0:
                yc = self.y[class_def]
                yc_med = np.median(yc)
                ycd = np.abs(yc - yc_med)
                adcm += sum(ycd)
        return adcm

    def get_gadf(self):
        """
        Goodness of absolute deviation of fit
        """
        adam = (np.abs(self.y - np.median(self.y))).sum()
        if adam == 0:   # array is invariant
            gadf = 1
        else:
            gadf = 1 - self.adcm / adam
        return gadf

    def _table_string(self, width=12, decimal=3):
        labels, largest = self.get_legend_classes(table=True)
        h1 = "Lower"
        h1 = h1.center(largest)
        h2 = " "
        h2 = h2.center(10)
        h3 = "Upper"
        h3 = h3.center(largest + 1)

        largest = "%d" % max(self.counts)
        largest = len(largest) + 15
        h4 = "Count"

        h4 = h4.rjust(largest)
        table = []
        header = h1 + h2 + h3 + h4
        table.append(header)
        table.append("=" * len(header))

        for i, label in enumerate(labels):
            left, right = label.split()
            if i == 0:
                left = " " * largest
                left += "   x[i] <= "
            else:
                left += " < x[i] <= "
            row = left + right
            cnt = "%d" % self.counts[i]
            cnt = cnt.rjust(largest)
            row += cnt
            table.append(row)
        name = self.name
        top = name.center(len(row))
        table.insert(0, top)
        table.insert(1, " ")
        table = "\n".join(table)
        return table

    def find_bin(self, x):
        """
        Sort input or inputs according to the current bin estimate

        Parameters
        ----------
        x       :   array or numeric
                    a value or array of values to fit within the estimated
                    bins

        Returns
        -------
        a bin index or array of bin indices that classify the input into one of
        the classifiers' bins.

        Note that this differs from similar functionality in
        numpy.digitize(x, classi.bins, right=True).

        This will always provide the closest bin, so data "outside" the classifier,
        above and below the max/min breaks, will be classified into the nearest bin.

        numpy.digitize returns k+1 for data greater than the greatest bin, but retains 0
        for data below the lowest bin.
        """
        x = np.asarray(x).flatten()
        right = np.digitize(x, self.bins, right=True)
        if right.max() == len(self.bins):
            right[right == len(self.bins)] = len(self.bins) - 1
        return right

    def get_legend_classes(self, fmt=FMT):
        """
        Format the strings for the classes on the legend


        Parameters
        ==========

        fmt : string
              formatting specification

        Returns
        =======
        classes: list
               k strings with class interval definitions
        """
        return _get_mpl_labels(self, fmt)

    def plot(
        self,
        gdf,
        border_color="lightgrey",
        border_width=0.10,
        title=None,
        legend=False,
        cmap="YlGnBu",
        axis_on=True,
        legend_kwds={"loc": "lower right", "fmt": FMT},
        file_name=None,
        dpi=600,
        ax=None,
    ):
        """
        Plot Mapclassiifer
        NOTE: Requires matplotlib, and implicitly requires geopandas
        dataframe as input.

        Parameters
        ---------
        gdf           : geopandas geodataframe
                        Contains the geometry column for the choropleth map
        border_color  : string, optional
                        matplotlib color string to use for polygon border
                        (Default: lightgrey)
        border_width  : float, optional
                        width of polygon boarder
                        (Default: 0.10)
        title         : string, optional
                        Title of map
                        (Default: None)
        cmap          : string, optional
                        matplotlib color string for color map to fill polygons
                        (Default: YlGn)
        axis_on       : boolean, optional
                        Show coordinate axes (default True)
                        (Default: True)
        legend_kwds   : dict, optional
                        options for ax.legend()
                        (Default: {"loc": "lower right", 'fmt':FMT})
        file_name     : string, optional
                        Name of file to save figure to.
                        (Default: None)
        dpi           : int, optional
                        Dots per inch for saved figure
                        (Default: 600)
        ax            : matplotlib axis, optional
                        axis on which to plot the choropleth.
                        (Default: None, so plots on the current figure)
        Returns
        -------
        f,ax        : tuple
                      matplotlib figure, axis on which the plot is made.


        Examples
        --------

        >>> import libpysal as lp
        >>> import geopandas
        >>> import mapclassify
        >>> gdf = geopandas.read_file(lp.examples.get_path("columbus.shp"))
        >>> q5 = mapclassify.Quantiles(gdf.CRIME)
        >>> q5.plot(gdf)  # doctest: +SKIP
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Mapclassify.plot depends on matplotlib.pyplot, and this was"
                "not able to be imported. \nInstall matplotlib to"
                "plot spatial classifier."
            )
        if ax is None:
            f = plt.figure()
            ax = plt.gca()
        else:
            f = plt.gcf()

        ax = gdf.assign(_cl=self.y).plot(
            column="_cl",
            ax=ax,
            cmap=cmap,
            edgecolor=border_color,
            linewidth=border_width,
            scheme=self.name,
            legend=legend,
            legend_kwds=legend_kwds,
        )
        if not axis_on:
            ax.axis("off")
        if title:
            f.suptitle(title)
        if file_name:
            plt.savefig(file_name, dpi=dpi)
        return f, ax


class HeadTailBreaks(MapClassifier):
    """
    Head/tail Breaks Map Classification for Heavy-tailed Distributions

    Parameters
    ----------
    y       : array
              (n,1), values to classify

    Attributes
    ----------
    yb      : array
              (n,1), bin ids for observations,
    bins    : array
              (k,1), the upper bounds of each class
    k       : int
              the number of classes
    counts  : array
              (k,1), the number of observations falling in each class

    Examples
    --------
    >>> import numpy as np
    >>> import mapclassify as mc
    >>> np.random.seed(10)
    >>> cal = mc.load_example()
    >>> htb = mc.HeadTailBreaks(cal)
    >>> htb.k
    3
    >>> list(htb.counts)
    [50, 7, 1]
    >>> htb.bins
    array([ 125.92810345,  811.26      , 4111.45      ])
    >>> np.random.seed(123456)
    >>> x = np.random.lognormal(3, 1, 1000)
    >>> htb = mc.HeadTailBreaks(x)
    >>> htb.bins
    array([ 32.26204423,  72.50205622, 128.07150107, 190.2899093 ,
           264.82847377, 457.88157946, 576.76046949])
    >>> list(htb.counts)
    [695, 209, 62, 22, 10, 1, 1]

    Notes
    -----
    Head/tail Breaks is a relatively new classification method developed
    for data with a heavy-tailed distribution.

    Implementation based on contributions by
    Alessandra Sozzi <alessandra.sozzi@gmail.com>.

    For theoretical details see :cite:`Jiang_2013`.

    """

    def __init__(self, y):
        MapClassifier.__init__(self, y)
        self.name = "HeadTailBreaks"

    def _set_bins(self):

        x = self.y.copy()
        bins = []
        bins = head_tail_breaks(x, bins)
        self.bins = np.array(bins)
        self.k = len(self.bins)


class EqualInterval(MapClassifier):
    """
    Equal Interval Classification

    Parameters
    ----------
    y : array
        (n,1), values to classify
    k : int
        number of classes required

    Attributes
    ----------
    yb      : array
              (n,1), bin ids for observations,
              each value is the id of the class the observation belongs to
              yb[i] = j  for j>=1  if bins[j-1] < y[i] <= bins[j], yb[i] = 0
              otherwise
    bins    : array
              (k,1), the upper bounds of each class
    k       : int
              the number of classes
    counts  : array
              (k,1), the number of observations falling in each class

    Examples
    --------
    >>> import mapclassify as mc
    >>> cal = mc.load_example()
    >>> ei = mc.EqualInterval(cal, k=5)
    >>> ei.k
    5
    >>> list(ei.counts)
    [57, 0, 0, 0, 1]
    >>> ei.bins
    array([ 822.394, 1644.658, 2466.922, 3289.186, 4111.45 ])

    Notes
    -----
    Intervals defined to have equal width:

    .. math::

        bins_j = min(y)+w*(j+1)

    with :math:`w=\\frac{max(y)-min(j)}{k}`
    """

    def __init__(self, y, k=K):
        """
        see class docstring

        """
        if min(y) == max(y):
            raise ValueError("Not enough unique values in array to form k classes.")
        self.k = k
        MapClassifier.__init__(self, y)
        self.name = "EqualInterval"

    def _set_bins(self):
        y = self.y
        k = self.k
        max_y = max(y)
        min_y = min(y)
        rg = max_y - min_y
        width = rg * 1.0 / k
        cuts = np.arange(min_y + width, max_y + width, width)
        if len(cuts) > self.k:  # handle overshooting
            cuts = cuts[0:k]
        cuts[-1] = max_y
        bins = cuts.copy()
        self.bins = bins


class Percentiles(MapClassifier):
    """
    Percentiles Map Classification

    Parameters
    ----------
    y    : array
           attribute to classify
    pct  : array
           percentiles default=[1,10,50,90,99,100]

    Attributes
    ----------
    yb     : array
             bin ids for observations (numpy array n x 1)
    bins   : array
             the upper bounds of each class (numpy array k x 1)
    k      : int
             the number of classes
    counts : int
             the number of observations falling in each class
             (numpy array k x 1)

    Examples
    --------
    >>> import mapclassify as mc
    >>> cal = mc.load_example()
    >>> p = mc.Percentiles(cal)
    >>> p.bins
    array([1.357000e-01, 5.530000e-01, 9.365000e+00, 2.139140e+02,
           2.179948e+03, 4.111450e+03])
    >>> list(p.counts)
    [1, 5, 23, 23, 5, 1]
    >>> p2 = mc.Percentiles(cal, pct = [50, 100])
    >>> p2.bins
    array([   9.365, 4111.45 ])
    >>> list(p2.counts)
    [29, 29]
    >>> p2.k
    2
    """

    def __init__(self, y, pct=[1, 10, 50, 90, 99, 100]):
        self.pct = pct
        MapClassifier.__init__(self, y)
        self.name = "Percentiles"

    def _set_bins(self):
        y = self.y
        pct = self.pct
        self.bins = np.array([stats.scoreatpercentile(y, p) for p in pct])
        self.k = len(self.bins)

    def update(self, y=None, inplace=False, **kwargs):
        """
        Add data or change classification parameters.

        Parameters
        ----------
        y   :   array
                    (n,1) array of data to classify
        inplace :   bool
                    whether to conduct the update in place or to return a copy
                    estimated from the additional specifications.

        Additional parameters provided in **kwargs are passed to the init
        function of the class. For documentation, check the class constructor.
        """
        kwargs.update({"pct": kwargs.pop("pct", self.pct)})
        if inplace:
            self._update(y, **kwargs)
        else:
            new = copy.deepcopy(self)
            new._update(y, **kwargs)
            return new


class BoxPlot(MapClassifier):
    """
    BoxPlot Map Classification

    Parameters
    ----------
    y     : array
            attribute to classify
    hinge : float
            multiplier for IQR

    Attributes
    ----------
    yb : array
        (n,1), bin ids for observations
    bins : array
          (n,1), the upper bounds of each class  (monotonic)
    k : int
        the number of classes
    counts : array
             (k,1), the number of observations falling in each class
    low_outlier_ids : array
        indices of observations that are low outliers
    high_outlier_ids : array
        indices of observations that are high outliers

    Notes
    -----

    The bins are set as follows::

        bins[0] = q[0]-hinge*IQR
        bins[1] = q[0]
        bins[2] = q[1]
        bins[3] = q[2]
        bins[4] = q[2]+hinge*IQR
        bins[5] = inf  (see Notes)

    where q is an array of the first three quartiles of y and
    IQR=q[2]-q[0]

    If q[2]+hinge*IQR > max(y) there will only be 5 classes and no high
    outliers, otherwise, there will be 6 classes and at least one high
    outlier.

    Examples
    --------

    >>> import mapclassify as mc
    >>> cal = mc.load_example()
    >>> bp = mc.BoxPlot(cal)
    >>> bp.bins
    array([-5.287625e+01,  2.567500e+00,  9.365000e+00,  3.953000e+01,
            9.497375e+01,  4.111450e+03])
    >>> list(bp.counts)
    [0, 15, 14, 14, 6, 9]
    >>> list(bp.high_outlier_ids)
    [0, 6, 18, 29, 33, 36, 37, 40, 42]
    >>> cal[bp.high_outlier_ids].values
    array([ 329.92,  181.27,  370.5 ,  722.85,  192.05,  110.74, 4111.45,
            317.11,  264.93])
    >>> bx = mc.BoxPlot(np.arange(100))
    >>> bx.bins
    array([-49.5 ,  24.75,  49.5 ,  74.25, 148.5 ])

    """

    def __init__(self, y, hinge=1.5):
        """
        Parameters
        ----------
        y : array (n,1)
            attribute to classify
        hinge : float
            multiple of inter-quartile range (default=1.5)
        """
        self.hinge = hinge
        MapClassifier.__init__(self, y)
        self.name = "BoxPlot"

    def _set_bins(self):
        y = self.y
        pct = [25, 50, 75, 100]
        bins = [stats.scoreatpercentile(y, p) for p in pct]
        iqr = bins[-2] - bins[0]
        self.iqr = iqr
        pivot = self.hinge * iqr
        left_fence = bins[0] - pivot
        right_fence = bins[-2] + pivot
        if right_fence < bins[-1]:
            bins.insert(-1, right_fence)
        else:
            bins[-1] = right_fence
        bins.insert(0, left_fence)
        self.bins = np.array(bins)
        self.k = len(bins)

    def _classify(self):
        MapClassifier._classify(self)
        self.low_outlier_ids = np.nonzero(self.yb == 0)[0]
        self.high_outlier_ids = np.nonzero(self.yb == 5)[0]

    def update(self, y=None, inplace=False, **kwargs):
        """
        Add data or change classification parameters.

        Parameters
        ----------
        y       :   array
                        (n,1) array of data to classify
        inplace     :   bool
                        whether to conduct the update in place or to return a
                        copy estimated from the additional specifications.

        Additional parameters provided in **kwargs are passed to the init
        function of the class. For documentation, check the class constructor.
        """
        kwargs.update({"hinge": kwargs.pop("hinge", self.hinge)})
        if inplace:
            self._update(y, **kwargs)
        else:
            new = copy.deepcopy(self)
            new._update(y, **kwargs)
            return new


class Quantiles(MapClassifier):
    """
    Quantile Map Classification

    Parameters
    ----------
    y : array
        (n,1), values to classify
    k : int
        number of classes required

    Attributes
    ----------
    yb      : array
              (n,1), bin ids for observations,
              each value is the id of the class the observation belongs to
              yb[i] = j  for j>=1  if bins[j-1] < y[i] <= bins[j], yb[i] = 0
              otherwise
    bins    : array
              (k,1), the upper bounds of each class
    k       : int
              the number of classes
    counts  : array
              (k,1), the number of observations falling in each class


    Examples
    --------
    >>> import mapclassify as mc
    >>> cal = mc.load_example()
    >>> q = mc.Quantiles(cal, k=5)
    >>> q.bins
    array([1.46400e+00, 5.79800e+00, 1.32780e+01, 5.46160e+01, 4.11145e+03])
    >>> list(q.counts)
    [12, 11, 12, 11, 12]
    """

    def __init__(self, y, k=K):
        self.k = k
        MapClassifier.__init__(self, y)
        self.name = "Quantiles"

    def _set_bins(self):
        y = self.y
        k = self.k
        self.bins = quantile(y, k=k)


class StdMean(MapClassifier):
    """
    Standard Deviation and Mean Map Classification

    Parameters
    ----------
    y         : array
                (n,1), values to classify
    multiples : array
                the multiples of the standard deviation to add/subtract from
                the sample mean to define the bins, default=[-2,-1,1,2]

    Attributes
    ----------

    yb      : array
              (n,1), bin ids for observations,
    bins    : array
              (k,1), the upper bounds of each class
    k       : int
              the number of classes
    counts  : array
              (k,1), the number of observations falling in each class

    Examples
    --------
    >>> import mapclassify as mc
    >>> cal = mc.load_example()
    >>> st = mc.StdMean(cal)
    >>> st.k
    5
    >>> st.bins
    array([-967.36235382, -420.71712519,  672.57333208, 1219.21856072,
           4111.45      ])
    >>> list(st.counts)
    [0, 0, 56, 1, 1]
    >>>
    >>> st3 = mc.StdMean(cal, multiples = [-3, -1.5, 1.5, 3])
    >>> st3.bins
    array([-1514.00758246,  -694.03973951,   945.8959464 ,  1765.86378936,
            4111.45      ])
    >>> list(st3.counts)
    [0, 0, 57, 0, 1]

    """

    def __init__(self, y, multiples=[-2, -1, 1, 2]):
        self.multiples = multiples
        MapClassifier.__init__(self, y)
        self.name = "StdMean"

    def _set_bins(self):
        y = self.y
        s = y.std(ddof=1)
        m = y.mean()
        cuts = [m + s * w for w in self.multiples]
        y_max = y.max()
        if cuts[-1] < y_max:
            cuts.append(y_max)
        self.bins = np.array(cuts)
        self.k = len(cuts)

    def update(self, y=None, inplace=False, **kwargs):
        """
        Add data or change classification parameters.

        Parameters
        ----------
        y   :   array
                    (n,1) array of data to classify
        inplace :   bool
                    whether to conduct the update in place or to return a copy
                    estimated from the additional specifications.

        Additional parameters provided in **kwargs are passed to the init
        function of the class. For documentation, check the class constructor.
        """
        kwargs.update({"multiples": kwargs.pop("multiples", self.multiples)})
        if inplace:
            self._update(y, **kwargs)
        else:
            new = copy.deepcopy(self)
            new._update(y, **kwargs)
            return new


class MaximumBreaks(MapClassifier):
    """
    Maximum Breaks Map Classification

    Parameters
    ----------
    y  : array
         (n, 1), values to classify

    k  : int
         number of classes required

    mindiff : float
              The minimum difference between class breaks

    Attributes
    ----------
    yb : array
         (n, 1), bin ids for observations
    bins : array
           (k, 1), the upper bounds of each class
    k    : int
           the number of classes
    counts : array
             (k, 1), the number of observations falling in each class (numpy
             array k x 1)

    Examples
    --------
    >>> import mapclassify as mc
    >>> cal = mc.load_example()
    >>> mb = mc.MaximumBreaks(cal, k=5)
    >>> mb.k
    5
    >>> mb.bins
    array([ 146.005,  228.49 ,  546.675, 2417.15 , 4111.45 ])
    >>> list(mb.counts)
    [50, 2, 4, 1, 1]

    """

    def __init__(self, y, k=5, mindiff=0):
        if min(y) == max(y):
            raise ValueError("Not enough unique values in array to form k classes.")
        self.k = k
        self.mindiff = mindiff
        MapClassifier.__init__(self, y)
        self.name = "MaximumBreaks"

    def _set_bins(self):
        xs = self.y.copy()
        k = self.k
        xs.sort()
        diffs = xs[1:] - xs[:-1]
        idxs = np.argsort(diffs)
        k1 = k - 1

        ud = np.unique(diffs)
        if len(ud) < k1:
            print("Insufficient number of unique diffs. Breaks are random.")
        mp = []
        for c in range(1, k):
            idx = idxs[-c]
            cp = (xs[idx] + xs[idx + 1]) / 2.0
            mp.append(cp)
        mp.append(xs[-1])
        mp.sort()
        self.bins = np.array(mp)

    def update(self, y=None, inplace=False, **kwargs):
        """
        Add data or change classification parameters.

        Parameters
        ----------
        y   :   array
                    (n,1) array of data to classify
        inplace :   bool
                    whether to conduct the update in place or to return a copy
                    estimated from the additional specifications.

        Additional parameters provided in **kwargs are passed to the init
        function of the class. For documentation, check the class constructor.
        """
        kwargs.update({"k": kwargs.pop("k", self.k)})
        kwargs.update({"mindiff": kwargs.pop("mindiff", self.mindiff)})
        if inplace:
            self._update(y, **kwargs)
        else:
            new = copy.deepcopy(self)
            new._update(y, **kwargs)
            return new


class NaturalBreaks(MapClassifier):
    """
    Natural Breaks Map Classification

    Parameters
    ----------
    y       : array
              (n,1), values to classify
    k       : int
              number of classes required

    initial : int, default: 10
              Number of initial solutions generated with different centroids.
              Best of initial results is returned.

    Attributes
    ----------

    yb      : array
              (n,1), bin ids for observations,
    bins    : array
              (k,1), the upper bounds of each class
    k       : int
              the number of classes
    counts  : array
              (k,1), the number of observations falling in each class

    Examples
    --------
    >>> import numpy as np
    >>> import mapclassify as mc
    >>> np.random.seed(123456)
    >>> cal = mc.load_example()
    >>> nb = mc.NaturalBreaks(cal, k=5)
    >>> nb.k
    5
    >>> list(nb.counts)
    [49, 3, 4, 1, 1]
    >>> nb.bins
    array([  75.29,  192.05,  370.5 ,  722.85, 4111.45])
    >>> x = np.array([1] * 50)
    >>> x[-1] = 20
    >>> nb = mc.NaturalBreaks(x, k=5)

    Warning: Not enough unique values in array to form k classes
    Warning: setting k to 2

    >>> nb.bins
    array([ 1, 20])
    >>> list(nb.counts)
    [49, 1]

    """

    def __init__(self, y, k=K, initial=10):
        self.k = k
        self.init = initial
        MapClassifier.__init__(self, y)
        self.name = "NaturalBreaks"

    def _set_bins(self):

        x = self.y.copy()
        k = self.k
        values = np.array(x)
        uv = np.unique(values)
        uvk = len(uv)
        if uvk < k:
            ms = "Warning: Not enough unique values in array to form k classes"
            Warn(ms, UserWarning)
            Warn("Warning: setting k to %d" % uvk, UserWarning)
            k = uvk
            uv.sort()
            # we set the bins equal to the sorted unique values and ramp k
            # downwards. no need to call kmeans.
            self.bins = uv
            self.k = k
        else:
            res0 = natural_breaks(x, k, init=self.init)
            self.bins = np.array(res0[-1])
            self.k = len(self.bins)

    def update(self, y=None, inplace=False, **kwargs):
        """
        Add data or change classification parameters.

        Parameters
        ----------
        y           :   array
                        (n,1) array of data to classify
        inplace     :   bool
                        whether to conduct the update in place or to return a
                        copy estimated from the additional specifications.

        Additional parameters provided in **kwargs are passed to the init
        function of the class. For documentation, check the class constructor.
        """
        kwargs.update({"k": kwargs.pop("k", self.k)})
        if inplace:
            self._update(y, **kwargs)
        else:
            new = copy.deepcopy(self)
            new._update(y, **kwargs)
            return new


class FisherJenks(MapClassifier):
    """
    Fisher Jenks optimal classifier - mean based

    Parameters
    ----------

    y : array
        (n,1), values to classify
    k : int, optional
        number of classes, defatuls to 5

    Attributes
    ----------

    yb      : array
              (n,1), bin ids for observations
    bins    : array
              (k,1), the upper bounds of each class
    k       : int
              the number of classes
    counts  : array
              (k,1), the number of observations falling in each class

    Examples
    --------

    >>> import mapclassify as mc
    >>> cal = mc.load_example()
    >>> fj = mc.FisherJenks(cal)
    >>> fj.adcm
    799.24
    >>> list(fj.bins)
    [75.29, 192.05, 370.5, 722.85, 4111.45]
    >>> list(fj.counts)
    [49, 3, 4, 1, 1]

    """

    def __init__(self, y, k=K):
        if not HAS_NUMBA:
            Warn("Numba not installed. Using slow pure python version.", UserWarning)

        nu = len(np.unique(y))
        if nu < k:
            raise ValueError("Fewer unique values than specified classes.")
        self.k = k
        MapClassifier.__init__(self, y)
        self.name = "FisherJenks"

    def _set_bins(self):
        x = np.sort(self.y).astype("f8")
        self.bins = _fisher_jenks_means(x, classes=self.k)


class FisherJenksSampled(MapClassifier):
    """
    Fisher Jenks optimal classifier - mean based using random sample

    Parameters
    ----------
    y      : array
             (n,1), values to classify
    k      : int
             number of classes required
    pct    : float
             The percentage of n that should form the sample
             If pct is specified such that n*pct > 1000, then
             pct = 1000./n, unless truncate is False
    truncate : boolean
               truncate pct in cases where pct * n > 1000., (Default True)

    Attributes
    ----------
    yb      : array
              (n,1), bin ids for observations
    bins    : array
              (k,1), the upper bounds of each class
    k       : int
              the number of classes
    counts  : array
              (k,1), the number of observations falling in each class

    Notes
    -----
    For theoretical details see :cite:`Rey_2016`.

    """

    def __init__(self, y, k=K, pct=0.10, truncate=True):
        self.k = k
        n = y.size

        if (pct * n > 1000) and truncate:
            pct = 1000.0 / n
        ids = np.random.randint(0, n, int(n * pct))
        y = np.asarray(y)
        yr = y[ids]
        yr[-1] = max(y)  # make sure we have the upper bound
        yr[0] = min(y)  # make sure we have the min
        self.original_y = y
        self.pct = pct
        self._truncated = truncate
        self.yr = yr
        self.yr_n = yr.size
        MapClassifier.__init__(self, yr)
        self.yb, self.counts = bin1d(y, self.bins)
        self.name = "FisherJenksSampled"
        self.y = y
        self._summary()  # have to recalculate summary stats

    def _set_bins(self):
        fj = FisherJenks(self.y, self.k)
        self.bins = fj.bins

    def update(self, y=None, inplace=False, **kwargs):
        """
        Add data or change classification parameters.

        Parameters
        ----------
        y           :   array
                        (n,1) array of data to classify
        inplace     :   bool
                        whether to conduct the update in place or to return a
                        copy estimated from the additional specifications.

        Additional parameters provided in **kwargs are passed to the init
        function of the class. For documentation, check the class constructor.
        """
        kwargs.update({"k": kwargs.pop("k", self.k)})
        kwargs.update({"pct": kwargs.pop("pct", self.pct)})
        kwargs.update({"truncate": kwargs.pop("truncate", self._truncated)})
        if inplace:
            self._update(y, **kwargs)
        else:
            new = copy.deepcopy(self)
            new._update(y, **kwargs)
            return new


class JenksCaspall(MapClassifier):
    """
    Jenks Caspall  Map Classification

    Parameters
    ----------
    y : array
        (n,1), values to classify
    k : int
        number of classes required

    Attributes
    ----------

    yb      : array
              (n,1), bin ids for observations,
    bins    : array
              (k,1), the upper bounds of each class
    k       : int
              the number of classes
    counts  : array
              (k,1), the number of observations falling in each class


    Examples
    --------
    >>> import mapclassify as mc
    >>> cal = mc.load_example()
    >>> jc = mc.JenksCaspall(cal, k=5)
    >>> jc.bins
    array([1.81000e+00, 7.60000e+00, 2.98200e+01, 1.81270e+02, 4.11145e+03])
    >>> list(jc.counts)
    [14, 13, 14, 10, 7]

    """

    def __init__(self, y, k=K):
        self.k = k
        MapClassifier.__init__(self, y)
        self.name = "JenksCaspall"

    def _set_bins(self):
        x = self.y.copy()
        k = self.k
        # start with quantiles
        q = quantile(x, k)
        solving = True
        xb, cnts = bin1d(x, q)
        # class means
        if x.ndim == 1:
            x.shape = (x.size, 1)
        n, k = x.shape
        xm = [np.median(x[xb == i]) for i in np.unique(xb)]
        xb0 = xb.copy()
        q = xm
        it = 0
        rk = list(range(self.k))
        while solving:
            xb = np.zeros(xb0.shape, int)
            d = abs(x - q)
            xb = d.argmin(axis=1)
            if (xb0 == xb).all():
                solving = False
            else:
                xb0 = xb
            it += 1
            q = np.array([np.median(x[xb == i]) for i in rk])
        cuts = np.array([max(x[xb == i]) for i in np.unique(xb)])
        cuts.shape = (len(cuts),)
        self.bins = cuts
        self.iterations = it


class JenksCaspallSampled(MapClassifier):
    """
    Jenks Caspall Map Classification using a random sample

    Parameters
    ----------

    y       : array
              (n,1), values to classify
    k       : int
              number of classes required
    pct     : float
              The percentage of n that should form the sample
              If pct is specified such that n*pct > 1000, then pct = 1000./n

    Attributes
    ----------

    yb      : array
              (n,1), bin ids for observations,
    bins    : array
              (k,1), the upper bounds of each class
    k       : int
              the number of classes
    counts  : array
              (k,1), the number of observations falling in each class


    Examples
    --------
    >>> import mapclassify as mc
    >>> import numpy as np
    >>> cal = mc.load_example()
    >>> np.random.seed(0)
    >>> x = np.random.random(100000)
    >>> jc = mc.JenksCaspall(x)
    >>> jcs = mc.JenksCaspallSampled(x)
    >>> jc.bins
    array([0.20108144, 0.4025151 , 0.60396127, 0.80302249, 0.99997795])
    >>> jcs.bins
    array([0.19978245, 0.40793025, 0.59253555, 0.78241472, 0.99997795])
    >>> list(jc.counts)
    [20286, 19951, 20310, 19708, 19745]
    >>> list(jcs.counts)
    [20147, 20633, 18591, 18857, 21772]

    # not for testing since we get different times on different hardware
    # just included for documentation of likely speed gains
    #>>> t1 = time.time(); jc = Jenks_Caspall(x); t2 = time.time()
    #>>> t1s = time.time(); jcs = Jenks_Caspall_Sampled(x); t2s = time.time()
    #>>> t2 - t1; t2s - t1s
    #1.8292930126190186
    #0.061631917953491211

    Notes
    -----
    This is intended for large n problems. The logic is to apply
    Jenks_Caspall to a random subset of the y space and then bin the
    complete vector y on the bins obtained from the subset. This would
    trade off some "accuracy" for a gain in speed.

    """

    def __init__(self, y, k=K, pct=0.10):
        self.k = k
        n = y.size
        if pct * n > 1000:
            pct = 1000.0 / n
        ids = np.random.randint(0, n, int(n * pct))
        y = np.asarray(y)
        yr = y[ids]
        yr[0] = max(y)  # make sure we have the upper bound
        self.original_y = y
        self.pct = pct
        self.yr = yr
        self.yr_n = yr.size
        MapClassifier.__init__(self, yr)
        self.yb, self.counts = bin1d(y, self.bins)
        self.name = "JenksCaspallSampled"
        self.y = y
        self._summary()  # have to recalculate summary stats

    def _set_bins(self):
        jc = JenksCaspall(self.y, self.k)
        self.bins = jc.bins
        self.iterations = jc.iterations

    def update(self, y=None, inplace=False, **kwargs):
        """
        Add data or change classification parameters.

        Parameters
        ----------
        y           :   array
                        (n,1) array of data to classify
        inplace     :   bool
                        whether to conduct the update in place or to return a
                        copy estimated from the additional specifications.

        Additional parameters provided in **kwargs are passed to the init
        function of the class. For documentation, check the class constructor.
        """
        kwargs.update({"k": kwargs.pop("k", self.k)})
        kwargs.update({"pct": kwargs.pop("pct", self.pct)})
        if inplace:
            self._update(y, **kwargs)
        else:
            new = copy.deepcopy(self)
            new._update(y, **kwargs)
            return new


class JenksCaspallForced(MapClassifier):
    """
    Jenks Caspall  Map Classification with forced movements

    Parameters
    ----------
    y : array
        (n,1), values to classify
    k : int
        number of classes required

    Attributes
    ----------
    yb      : array
              (n,1), bin ids for observations
    bins    : array
              (k,1), the upper bounds of each class
    k       : int
              the number of classes
    counts  : array
              (k,1), the number of observations falling in each class


    Examples
    --------
    >>> import mapclassify as mc
    >>> cal = mc.load_example()
    >>> jcf = mc.JenksCaspallForced(cal, k=5)
    >>> jcf.k
    5
    >>> jcf.bins
    array([1.34000e+00, 5.90000e+00, 1.67000e+01, 5.06500e+01, 4.11145e+03])
    >>> list(jcf.counts)
    [12, 12, 13, 9, 12]
    >>> jcf4 = mc.JenksCaspallForced(cal, k=4)
    >>> jcf4.k
    4
    >>> jcf4.bins
    array([2.51000e+00, 8.70000e+00, 3.66800e+01, 4.11145e+03])
    >>> list(jcf4.counts)
    [15, 14, 14, 15]
    """

    def __init__(self, y, k=K):
        if min(y) == max(y):
            raise ValueError("Not enough unique values in array to form k classes.")
        self.k = k
        MapClassifier.__init__(self, y)
        self.name = "JenksCaspallForced"

    def _set_bins(self):
        x = self.y.copy()
        k = self.k
        q = quantile(x, k)
        solving = True
        xb, cnt = bin1d(x, q)
        # class means
        if x.ndim == 1:
            x.shape = (x.size, 1)
        n, tmp = x.shape
        xm = [x[xb == i].mean() for i in np.unique(xb)]
        q = xm
        xbar = np.array([xm[xbi] for xbi in xb])
        xbar.shape = (n, 1)
        ss = x - xbar
        ss *= ss
        ss = sum(ss)
        down_moves = up_moves = 0
        solving = True
        it = 0
        while solving:
            # try upward moves first
            moving_up = True
            while moving_up:
                class_ids = np.unique(xb)
                nk = [sum(xb == j) for j in class_ids]
                candidates = nk[:-1]
                i = 0
                up_moves = 0
                while candidates:
                    nki = candidates.pop(0)
                    if nki > 1:
                        ids = np.nonzero(xb == class_ids[i])
                        mover = max(ids[0])
                        tmp = xb.copy()
                        tmp[mover] = xb[mover] + 1
                        tm = [x[tmp == j].mean() for j in np.unique(tmp)]
                        txbar = np.array([tm[xbi] for xbi in tmp])
                        txbar.shape = (n, 1)
                        tss = x - txbar
                        tss *= tss
                        tss = sum(tss)
                        if tss < ss:
                            xb = tmp
                            ss = tss
                            candidates = []
                            up_moves += 1
                    i += 1
                if not up_moves:
                    moving_up = False
            moving_down = True
            while moving_down:
                class_ids = np.unique(xb)
                nk = [sum(xb == j) for j in class_ids]
                candidates = nk[1:]
                i = 1
                down_moves = 0
                while candidates:
                    nki = candidates.pop(0)
                    if nki > 1:
                        ids = np.nonzero(xb == class_ids[i])
                        mover = min(ids[0])
                        mover_class = xb[mover]
                        target_class = mover_class - 1
                        tmp = xb.copy()
                        tmp[mover] = target_class
                        tm = [x[tmp == j].mean() for j in np.unique(tmp)]
                        txbar = np.array([tm[xbi] for xbi in tmp])
                        txbar.shape = (n, 1)
                        tss = x - txbar
                        tss *= tss
                        tss = sum(tss)
                        if tss < ss:
                            xb = tmp
                            ss = tss
                            candidates = []
                            down_moves += 1
                    i += 1
                if not down_moves:
                    moving_down = False
            if not up_moves and not down_moves:
                solving = False
            it += 1
        cuts = [max(x[xb == c]) for c in np.unique(xb)]
        cuts = np.reshape(np.array(cuts), (k,))
        self.bins = cuts
        self.iterations = it


class UserDefined(MapClassifier):
    """
    User Specified Binning

    Parameters
    ----------
    y    : array
           (n,1), values to classify
    bins : array
           (k,1), upper bounds of classes (have to be monotically increasing)

    lowest  : float (optional)
           scalar minimum value of lowest class. Default is to set the minimum
           to -inf if  y.min() > first upper bound, otherwise minimum is set to
           y.min(). lowest will override the default



    Attributes
    ----------
    yb      : array
              (n,1), bin ids for observations,
    bins    : array
              (k,1), the upper bounds of each class
    k       : int
              the number of classes
    counts  : array
              (k,1), the number of observations falling in each class


    Examples
    --------
    >>> import mapclassify as mc
    >>> cal = mc.load_example()
    >>> bins = [20, max(cal)]
    >>> bins
    [20, 4111.45]
    >>> ud = mc.UserDefined(cal, bins)
    >>> list(ud.bins)
    [20.0, 4111.45]
    >>> list(ud.counts)
    [37, 21]
    >>> bins = [20, 30]
    >>> ud = mc.UserDefined(cal, bins)
    >>> list(ud.bins)
    [20.0, 30.0, 4111.45]
    >>> list(ud.counts)
    [37, 4, 17]

    Notes
    -----
    If upper bound of user bins does not exceed max(y) we append an
    additional bin.

    """

    def __init__(self, y, bins, lowest=None):
        if bins[-1] < max(y):
            bins = np.append(bins, max(y))
        self.lowest = lowest
        self.k = len(bins)
        self.bins = np.array(bins)
        self.y = y
        MapClassifier.__init__(self, y)
        self.name = "UserDefined"

    def _set_bins(self):
        pass

    def _update(self, y=None, bins=None):
        if y is not None:
            if hasattr(y, "values"):
                y = y.values
            y = np.append(y.flatten(), self.y)
        else:
            y = self.y
        if bins is None:
            bins = self.bins
        self.__init__(y, bins)

    def update(self, y=None, inplace=False, **kwargs):
        """
        Add data or change classification parameters.

        Parameters
        ----------
        y           :   array
                        (n,1) array of data to classify
        inplace     :   bool
                        whether to conduct the update in place or to return a
                        copy estimated from the additional specifications.

        Additional parameters provided in **kwargs are passed to the init
        function of the class. For documentation, check the class constructor.
        """
        bins = kwargs.pop("bins", self.bins)
        if inplace:
            self._update(y=y, bins=bins, **kwargs)
        else:
            new = copy.deepcopy(self)
            new._update(y, bins, **kwargs)
            return new

    # We have to override the plot method for additional kwargs for UserDefined
    def plot(
        self,
        gdf,
        border_color="lightgrey",
        border_width=0.10,
        title=None,
        legend=False,
        cmap="YlGnBu",
        axis_on=True,
        legend_kwds={"loc": "lower right", "fmt": FMT},
        file_name=None,
        dpi=600,
        ax=None,
    ):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Mapclassify.plot depends on matplotlib.pyplot, and this was"
                "not able to be imported. \nInstall matplotlib to"
                "plot spatial classifier."
            )
        if ax is None:
            f = plt.figure()
            ax = plt.gca()
        else:
            f = plt.gcf()

        if "fmt" in legend_kwds:
            legend_kwds.pop("fmt")

        ax = gdf.assign(_cl=self.y).plot(
            column="_cl",
            ax=ax,
            cmap=cmap,
            edgecolor=border_color,
            linewidth=border_width,
            scheme=self.name,
            legend=legend,
            legend_kwds=legend_kwds,
            classification_kwds={"bins": self.bins},  # for UserDefined
        )
        if not axis_on:
            ax.axis("off")
        if title:
            f.suptitle(title)
        if file_name:
            plt.savefig(file_name, dpi=dpi)
        return f, ax


class MaxP(MapClassifier):
    """
    MaxP Map Classification. Based on Max-p regionalization algorithm.

    Parameters
    ----------

    y : array
        ``(n,1)``, values to classify
    k : int
        number of classes required
    initial : int
        number of initial solutions to use prior to swapping
    seed1 : int
        Random state for initial building process. Default is ``0``.
    seed2 : int
        Random state for swapping process. Default is ``1``.

    Attributes
    ----------

    yb : array
        ``(n,1)``, bin ids for observations
    bins : array
        ``(k,1)``, the upper bounds of each class
    k : int
        the number of classes
    counts  : array
        ``(k,1)``, the number of observations falling in each class

    Examples
    --------

    >>> import mapclassify as mc
    >>> cal = mc.load_example()
    >>> mp = mc.MaxP(cal)
    >>> mp.bins
    array([3.16000e+00, 1.26300e+01, 1.67000e+01, 2.04700e+01, 4.11145e+03])

    >>> list(mp.counts)
    [18, 16, 3, 1, 20]

    """

    def __init__(self, y, k=K, initial=1000, seed1=0, seed2=1):
        if min(y) == max(y):
            raise ValueError("Not enough unique values in array to form k classes.")
        self.k = k
        self.initial = initial
        self.seed1 = seed1
        self.seed2 = seed2
        MapClassifier.__init__(self, y)
        self.name = "MaxP"

    def _set_bins(self):
        x = self.y.copy()
        k = self.k
        q = quantile(x, k)
        if x.ndim == 1:
            x.shape = (x.size, 1)
        n, tmp = x.shape
        x.sort(axis=0)
        # find best of initial solutions
        solution = 0
        best_tss = x.var() * x.shape[0]
        tss_all = np.zeros((self.initial, 1))
        while solution < self.initial:
            remaining = list(range(n))
            seeds = [
                np.nonzero(di == min(di))[0][0] for di in [np.abs(x - qi) for qi in q]
            ]
            np.random.seed(self.seed1)
            rseeds = np.random.permutation(list(range(k))).tolist()
            [remaining.remove(seed) for seed in seeds]
            self.classes = classes = []
            [classes.append([seed]) for seed in seeds]
            while rseeds:
                seed_id = rseeds.pop()
                current = classes[seed_id]
                growing = True
                while growing:
                    current = classes[seed_id]
                    low = current[0]
                    high = current[-1]
                    left = low - 1
                    right = high + 1
                    move_made = False
                    if left in remaining:
                        current.insert(0, left)
                        remaining.remove(left)
                        move_made = True
                    if right in remaining:
                        current.append(right)
                        remaining.remove(right)
                        move_made = True
                    if move_made:
                        classes[seed_id] = current
                    else:
                        growing = False
            tss = _fit(self.y, classes)
            tss_all[solution] = tss
            if tss < best_tss:
                best_solution = classes
                best_it = solution
                best_tss = tss
            solution += 1
        classes = best_solution
        self.best_it = best_it
        self.tss = best_tss
        self.a2c = a2c = {}
        self.tss_all = tss_all
        for r, cl in enumerate(classes):
            for a in cl:
                a2c[a] = r
        swapping = True
        while swapping:
            np.random.seed(self.seed2)
            rseeds = np.random.permutation(list(range(k))).tolist()
            total_moves = 0
            while rseeds:
                id = rseeds.pop()
                growing = True
                total_moves = 0
                while growing:
                    target = classes[id]
                    left = target[0] - 1
                    right = target[-1] + 1
                    n_moves = 0
                    if left in a2c:
                        left_class = classes[a2c[left]]
                        if len(left_class) > 1:
                            a = left_class[-1]
                            if self._swap(left_class, target, a):
                                target.insert(0, a)
                                left_class.remove(a)
                                a2c[a] = id
                                n_moves += 1
                    if right in a2c:
                        right_class = classes[a2c[right]]
                        if len(right_class) > 1:
                            a = right_class[0]
                            if self._swap(right_class, target, a):
                                target.append(a)
                                right_class.remove(a)
                                n_moves += 1
                                a2c[a] = id
                    if not n_moves:
                        growing = False
                total_moves += n_moves
            if not total_moves:
                swapping = False
        xs = self.y.copy()
        xs.sort()
        self.bins = np.array([xs[cl][-1] for cl in classes])

    def _ss(self, class_def):
        """calculates sum of squares for a class"""
        yc = self.y[class_def]
        css = yc - yc.mean()
        css *= css
        return sum(css)

    def _swap(self, class1, class2, a):
        """evaluate cost of moving a from class1 to class2"""
        ss1 = self._ss(class1)
        ss2 = self._ss(class2)
        tss1 = ss1 + ss2
        class1c = copy.copy(class1)
        class2c = copy.copy(class2)
        class1c.remove(a)
        class2c.append(a)
        ss1 = self._ss(class1c)
        ss2 = self._ss(class2c)
        tss2 = ss1 + ss2
        if tss1 < tss2:
            return False
        else:
            return True

    '''
    def update(self, y=None, inplace=False, **kwargs):
        """
        Add data or change classification parameters.

        Parameters
        ----------
        y           :   array
                        (n,1) array of data to classify
        inplace     :   bool
                        whether to conduct the update in place or to return a
                        copy estimated from the additional specifications.

        Additional parameters provided in **kwargs are passed to the init
        function of the class. For documentation, check the class constructor.
        """
        kwargs.update({"initial": kwargs.pop("initial", self.initial)})
        if inplace:
            self._update(y, bins, **kwargs)
        else:
            new = copy.deepcopy(self)
            new._update(y, bins, **kwargs)
            return new
    '''


def _fit(y, classes):
    """Calculate the total sum of squares for a vector y classified into
    classes

    Parameters
    ----------
    y : array
        (n,1), variable to be classified

    classes : array
              (k,1), integer values denoting class membership

    """
    tss = 0
    for class_def in classes:
        yc = y[class_def]
        css = yc - yc.mean()
        css *= css
        tss += sum(css)
    return tss


kmethods = {}
kmethods["Quantiles"] = Quantiles
kmethods["FisherJenks"] = FisherJenks
kmethods["NaturalBreaks"] = NaturalBreaks
kmethods["MaximumBreaks"] = MaximumBreaks


def gadf(y, method="Quantiles", maxk=15, pct=0.8):
    r"""
    Evaluate the Goodness of Absolute Deviation Fit of a Classifier
    Finds the minimum value of k for which gadf>pct

    Parameters
    ----------

    y      : array
             (n, 1) values to be classified
    method : {'Quantiles, 'Fisher_Jenks', 'Maximum_Breaks', 'Natrual_Breaks'}
    maxk   : int
             maximum value of k to evaluate
    pct    : float
             The percentage of GADF to exceed

    Returns
    -------
    k : int
        number of classes
    cl : object
         instance of the classifier at k
    gadf : float
           goodness of absolute deviation fit

    Examples
    --------
    >>> import mapclassify as mc
    >>> cal = mc.load_example()
    >>> qgadf = mc.classifiers.gadf(cal)
    >>> qgadf[0]
    15
    >>> qgadf[-1]
    0.3740257590909283

    Quantiles fail to exceed 0.80 before 15 classes. If we lower the bar to
    0.2 we see quintiles as a result

    >>> qgadf2 = mc.classifiers.gadf(cal, pct = 0.2)
    >>> qgadf2[0]
    5
    >>> qgadf2[-1]
    0.21710231966462412
    >>>

    Notes
    -----

    The GADF is defined as:

        .. math::

            GADF = 1 - \sum_c \sum_{i \in c}
                   |y_i - y_{c,med}|  / \sum_i |y_i - y_{med}|

        where :math:`y_{med}` is the global median and :math:`y_{c,med}` is
        the median for class :math:`c`.

    See Also
    --------
    KClassifiers
    """

    y = np.array(y)
    adam = (np.abs(y - np.median(y))).sum()
    for k in range(2, maxk + 1):
        cl = kmethods[method](y, k)
        gadf = 1 - cl.adcm / adam
        if gadf > pct:
            break
    return (k, cl, gadf)


class KClassifiers(object):
    """
    Evaluate all k-classifers and pick optimal based on k and GADF

    Parameters
    ----------
    y      : array
             (n,1), values to be classified
    pct    : float
             The percentage of GADF to exceed

    Attributes
    ----------
    best   :  object
              instance of the optimal MapClassifier
    results : dictionary
              keys are classifier names, values are the MapClassifier
              instances with the best pct for each classifer

    Examples
    --------
    >>> import mapclassify as mc
    >>> cal = mc.load_example()
    >>> ks = mc.classifiers.KClassifiers(cal)
    >>> ks.best.name
    'FisherJenks'
    >>> ks.best.k
    4
    >>> ks.best.gadf
    0.8481032719908105

    Notes
    -----
    This can be used to suggest a classification scheme.

    See Also
    --------
    gadf

    """

    def __init__(self, y, pct=0.8):
        results = {}
        best = gadf(y, "FisherJenks", maxk=len(y) - 1, pct=pct)
        pct0 = best[0]
        k0 = best[-1]
        keys = list(kmethods.keys())
        keys.remove("FisherJenks")
        results["FisherJenks"] = best
        for method in keys:
            results[method] = gadf(y, method, maxk=len(y) - 1, pct=pct)
            k1 = results[method][0]
            pct1 = results[method][-1]
            if (k1 < k0) or (k1 == k0 and pct0 < pct1):
                best = results[method]
                k0 = k1
                pct0 = pct1
        self.results = results
        self.best = best[1]
