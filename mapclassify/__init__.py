import contextlib
from importlib.metadata import PackageNotFoundError, version

from . import util
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
    PrettyBreaks,
    Quantiles,
    StdMean,
    UserDefined,
    gadf,
    load_example,
)
from .greedy import greedy
from .pooling import Pooled

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("mapclassify")
