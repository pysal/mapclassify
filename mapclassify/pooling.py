import numpy as np

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


class Pooled(object):
    """Applying global binning across columns

    Parameters
    ----------

    Y : array
        (n, m), values to classify, with m>1

    classifier : string
                Name of mapclassify.classifier to apply

    **kwargs : dict
              additional keyword arguments for classifier

    Attributes
    ----------

    global_classifier : MapClassifier
               Instance of the pooled classifier defined as the classifier
               applied to the union of the columns.

    col_classifier : list
               Elements are MapClassifier instances with the pooled classifier
               applied to the associated column of Y.

    Examples
    --------
    >>> import numpy as np
    >>> import mapclassify as mc
    >>> n = 20
    >>> data = np.array([np.arange(n)+i*n for i in range(1,4)]).T
    >>> res = mc.Pooled(data)
    >>> list(res.col_classifiers[0].counts)
    [12, 8, 0, 0, 0]
    >>> list(res.col_classifiers[1].counts)
    [0, 4, 12, 4, 0]
    >>> list(res.col_classifiers[2].counts)
    [0, 0, 0, 8, 12]
    >>> list(res.global_classifier.counts)
    [12, 12, 12, 12, 12]
    >>> res.global_classifier.bins == res.col_classifiers[0].bins
    array([ True,  True,  True,  True,  True])
    >>> res.global_classifier.bins
    array([31.8, 43.6, 55.4, 67.2, 79. ])
    """

    def __init__(self, Y, classifier="Quantiles", **kwargs):
        self.__dict__.update(kwargs)
        Y = np.asarray(Y)
        n, cols = Y.shape
        y = np.reshape(Y, (-1, 1), order="f")
        ymin = y.min()
        method = classifier.lower()
        if method not in dispatcher:
            print(f"{method} not a valid classifier.")
            return None
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
