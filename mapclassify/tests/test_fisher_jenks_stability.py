import numpy as np

from mapclassify import FisherJenks


def test_fisher_jenks_is_translation_invariant():
    values = np.array([0, 1, 2, 50, 51, 52, 100, 101, 102], dtype=float)
    expected = FisherJenks(values, k=3)

    for offset in (273.15, 1_000_000.0):
        shifted = FisherJenks(values + offset, k=3)
        np.testing.assert_array_equal(shifted.yb, expected.yb)
        np.testing.assert_allclose(shifted.bins - offset, expected.bins, atol=1e-10)
