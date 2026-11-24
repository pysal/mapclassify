"""Every classifier must cope with tied data: no hangs, no crashes, coherent k.

See GLOSSARY.md (Tie, k) and docs/adr/0001-ties-never-split.md.
"""

import signal
import warnings

import numpy
import pytest

import mapclassify

K = 5
TIMEOUT = 20  # seconds; a classifier that has not finished by then is hung

_rng = numpy.random.default_rng(0)
DATASETS = {
    # one large mass point plus a continuous tail
    "mass_point": numpy.r_[numpy.zeros(70), _rng.lognormal(size=30)],
    # fewer distinct values than requested classes
    "three_values": numpy.repeat([1.0, 2.0, 3.0], [50, 30, 20]),
    # a single distinct value
    "constant": numpy.ones(20),
}

WITH_K = [
    "Quantiles",
    "EqualInterval",
    "MaximumBreaks",
    "NaturalBreaks",
    "FisherJenks",
    "FisherJenksSampled",
    "JenksCaspall",
    "JenksCaspallSampled",
    "JenksCaspallForced",
    "PrettyBreaks",
    "MaxP",
]
WITHOUT_K = ["Percentiles", "BoxPlot", "StdMean", "HeadTailBreaks"]


class _HungError(Exception):
    pass


def _fit(name, y):
    cls = getattr(mapclassify, name)
    kwargs = {"k": K} if name in WITH_K else {}
    use_alarm = hasattr(signal, "SIGALRM")
    old = None
    if use_alarm:

        def _raise(*_):
            raise _HungError(f"{name} did not finish within {TIMEOUT}s")

        old = signal.signal(signal.SIGALRM, _raise)
        signal.alarm(TIMEOUT)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            numpy.random.seed(12345)
            return cls(y, **kwargs)
    finally:
        if use_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)


@pytest.mark.parametrize("data", list(DATASETS))
@pytest.mark.parametrize("name", WITH_K + WITHOUT_K)
def test_tied_data(name, data):
    y = DATASETS[data]
    try:
        c = _fit(name, y)
    except ValueError as e:
        # A deliberate refusal is allowed (until the `ties=` policy lands),
        # but it must say why; IndexError and friends are bugs.
        assert "unique" in str(e).lower(), f"unhelpful ValueError: {e}"
        return

    # k is the number of classes actually produced
    assert c.k == len(c.bins) == len(c.counts)
    assert len(c.classes) == c.k
    # every observation is classified exactly once
    assert c.counts.sum() == y.size
    assert numpy.all(numpy.diff(c.bins) >= 0)
    # tied observations are never split across classes
    for v in numpy.unique(y):
        assert len(numpy.unique(c.yb[y == v])) == 1
    # k never exceeds the number of distinct values... except where
    # empty or zero-width classes are produced by design (positional
    # schemes, equal-width intervals, pretty breaks).
    if name in ("NaturalBreaks", "FisherJenks", "JenksCaspall"):
        assert c.k <= len(numpy.unique(y))


@pytest.mark.xfail(
    strict=True,
    reason="Quantiles interpolates breaks between observed values, creating an "
    "empty class (e.g. 2.2); fixed by switching to observed-value breaks.",
)
def test_quantiles_k_at_most_distinct_values():
    y = DATASETS["three_values"]
    c = _fit("Quantiles", y)
    assert c.k <= len(numpy.unique(y))


def test_update_reuses_requested_k():
    # A collapse to fewer classes must not stick: updating with richer data
    # asks for the originally requested k again.
    y = DATASETS["three_values"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q = mapclassify.Quantiles(y, k=K)
    assert q.k < K
    assert q.k_requested == K
    q2 = q.update(numpy.arange(100, 200, dtype=float))
    assert q2.k == K
