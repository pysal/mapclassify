import numpy as np
from .classifiers import (BoxPlot, EqualInterval, FisherJenks,
                          FisherJenksSampled, Quantiles, UserDefined,
                          NaturalBreaks, MaximumBreaks, MaxP, StdMean)

__all__ = ["Pooled"]

dispatcher = {'boxplot': BoxPlot,
              'equalinterval': EqualInterval,
              'fisherjenks': FisherJenks,
              'fisherjenkssampled': FisherJenksSampled,
              'quantiles': Quantiles,
              'maximumbreaks': MaximumBreaks,
              'stdmean': StdMean
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

    """
    def __init__(self, Y,  classifier='Quantiles', **kwargs):
        self.__dict__.update(kwargs)
        Y = np.asarray(Y)
        n, cols = Y.shape
        y = np.reshape(Y, (-1, 1), order='f')
        method = classifier.lower()
        if method not in dispatcher:
            print(f'{method} not a valid classifier.')
            return None
        global_classifier = dispatcher[method](y, **kwargs)
        #self.k = global_classifier.k
        col_classifiers = []
        name = f'Pooled {classifier}'
        for c in range(cols):
            res = UserDefined(Y[:, c], bins=global_classifier.bins)
            res.name = name
            col_classifiers.append(res)
        self.col_classifiers = col_classifiers
        self.global_classifier = global_classifier
        self._summary()

    def _summary(self):
        yb = self.global_classifier.yb
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


