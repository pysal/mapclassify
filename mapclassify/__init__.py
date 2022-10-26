from . import _version
from ._classify_API import classify
from .classifiers import (
    CLASSIFIERS,
    BoxPlot,
    EqualInterval,
    FisherJenks,
    FisherJenksSampled,
    HeadTailBreaks,
    JenksCaspall,
    JenksCaspallForced,
    JenksCaspallSampled,
    KClassifiers,
    MaximumBreaks,
    MaxP,
    NaturalBreaks,
    Percentiles,
    Quantiles,
    StdMean,
    UserDefined,
    gadf,
    load_example,
)
from .greedy import greedy
from .pooling import Pooled

__version__ = _version.get_versions()["version"]
