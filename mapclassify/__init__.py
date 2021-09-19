from .classifiers import (
    BoxPlot,
    EqualInterval,
    FisherJenks,
    FisherJenksSampled,
    HeadTailBreaks,
    JenksCaspall,
    JenksCaspallForced,
    JenksCaspallSampled,
    MaxP,
    MaximumBreaks,
    NaturalBreaks,
    Quantiles,
    Percentiles,
    StdMean,
    UserDefined,
    load_example,
    gadf,
    KClassifiers,
    CLASSIFIERS,
)

from .pooling import Pooled
from .greedy import greedy

from ._classify_API import classify

from . import _version
__version__ = _version.get_versions()['version']
