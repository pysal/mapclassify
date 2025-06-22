import contextlib
from importlib.metadata import PackageNotFoundError, version

from . import legendgram, util
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
from .value_by_alpha import vba_choropleth

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("mapclassify")
