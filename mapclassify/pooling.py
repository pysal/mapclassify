import numpy

from .classifiers import (
    BoxPlot,
    EqualInterval,
    FisherJenks,
    FisherJenksSampled,
    MaximumBreaks,
    Quantiles,
    StdMean,
    UserDefined,
)

__all__ = ["Pooled"]

dispatcher = {
    "boxplot": BoxPlot,
    "equalinterval": EqualInterval,
    "fisherjenks": FisherJenks,
    "fisherjenkssampled": FisherJenksSampled,
    "quantiles": Quantiles,
    "maximumbreaks": MaximumBreaks,
    "stdmean": StdMean,
    "userdefined": UserDefined,
}


class Pooled:
    """Applying global binning across columns.

    Parameters
    ----------

    Y : numpy.array
        :math:`(n, m)`, values to classify, with :math:`m>1`.
    classifier : str (default 'Quantiles')
        Name of ``mapclassify.classifier`` to apply.
    **kwargs : dict
        Additional keyword arguments for classifier.

    Attributes
    ----------

    global_classifier : mapclassify.classifiers.MapClassifier
        Instance of the pooled classifier defined as the classifier
        applied to the union of the columns.
    col_classifier : list
        Elements are ``MapClassifier`` instances with the pooled classifier
        applied to the associated column of ``Y``.

    Examples
    --------

    >>> import mapclassify
    >>> import numpy
    >>> n = 20
    >>> data = numpy.array([numpy.arange(n)+i*n for i in range(1,4)]).T
    >>> res = mapclassify.Pooled(data)

    >>> res.col_classifiers[0].counts.tolist()
    [12, 8, 0, 0, 0]

    >>> res.col_classifiers[1].counts.tolist()
    [0, 4, 12, 4, 0]

    >>> res.col_classifiers[2].counts.tolist()
    [0, 0, 0, 8, 12]

    >>> res.global_classifier.counts.tolist()
    [12, 12, 12, 12, 12]

    >>> res.global_classifier.bins == res.col_classifiers[0].bins
    array([ True,  True,  True,  True,  True])

    >>> res.global_classifier.bins
    array([31.8, 43.6, 55.4, 67.2, 79. ])

    """

    def __init__(self, Y, classifier="Quantiles", **kwargs):  # noqa: N803
        method = classifier.lower()
        valid_methods = list(dispatcher.keys())
        if method not in valid_methods:
            raise ValueError(
                f"'{classifier}' not a valid classifier. "
                f"Currently supported classifiers: {valid_methods}"
            )

        self.__dict__.update(kwargs)
        _y = numpy.asarray(Y)
        n, cols = _y.shape
        y = numpy.reshape(_y, (-1, 1), order="f")
        ymin = y.min()
        global_classifier = dispatcher[method](y, **kwargs)
        # self.k = global_classifier.k
        col_classifiers = []
        name = f"Pooled {classifier}"
        for c in range(cols):
            res = UserDefined(Y[:, c], bins=global_classifier.bins, lowest=ymin)
            res.name = name
            col_classifiers.append(res)
        self.col_classifiers = col_classifiers
        self.global_classifier = global_classifier
        self._summary()

    def _summary(self):
        self.classes = self.global_classifier.classes
        self.tss = self.global_classifier.tss
        self.adcm = self.global_classifier.adcm
        self.gadf = self.global_classifier.gadf

    def __str__(self):
        s = "Pooled Classifier"
        rows = [s]
        for c in self.col_classifiers:
            rows.append(c.table())
        return "\n\n".join(rows)

    def __repr__(self):
        return self.__str__()
