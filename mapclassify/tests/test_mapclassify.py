import types

import numpy
import pytest

from ..classifiers import *
from ..classifiers import bin, bin1d, binC, load_example
from ..pooling import Pooled

RTOL = 0.0001


class TestQuantile:
    def test_quantile(self):
        y = numpy.arange(1000)
        expected = numpy.array([333.0, 666.0, 999.0])
        numpy.testing.assert_almost_equal(expected, quantile(y, k=3))

    def test_quantile_k4(self):
        x = numpy.arange(1000)
        qx = quantile(x, k=4)
        expected = numpy.array([249.75, 499.5, 749.25, 999.0])
        numpy.testing.assert_array_almost_equal(expected, qx)

    def test_quantile_k(self):
        y = numpy.random.random(1000)
        for k in range(5, 10):
            numpy.testing.assert_almost_equal(k, len(quantile(y, k)))
            assert k == len(quantile(y, k))


class TestUpdate:
    def setup_method(self):
        numpy.random.seed(4414)
        self.data = numpy.random.normal(0, 10, size=10)
        self.new_data = numpy.random.normal(0, 10, size=4)

    def test_update(self):
        # Quantiles
        quants = Quantiles(self.data, k=3)
        known_yb = numpy.array([0, 1, 0, 1, 0, 2, 0, 2, 1, 2])
        numpy.testing.assert_allclose(quants.yb, known_yb, rtol=RTOL)

        new_yb = quants.update(self.new_data, k=4).yb
        known_new_yb = numpy.array([0, 3, 1, 0, 1, 2, 0, 2, 1, 3, 0, 3, 2, 3])
        numpy.testing.assert_allclose(new_yb, known_new_yb, rtol=RTOL)

        # User-Defined
        ud = UserDefined(self.data, [-20, 0, 5, 20])
        known_yb = numpy.array([1, 2, 1, 1, 1, 2, 0, 2, 1, 3])
        numpy.testing.assert_allclose(ud.yb, known_yb, rtol=RTOL)

        new_yb = ud.update(self.new_data).yb
        known_new_yb = numpy.array([1, 3, 1, 1, 1, 2, 1, 1, 1, 2, 0, 2, 1, 3])
        numpy.testing.assert_allclose(new_yb, known_new_yb, rtol=RTOL)

        # Fisher-Jenks Sampled
        fjs = FisherJenksSampled(self.data, k=3, pct=70)
        known_yb = numpy.array([1, 2, 0, 1, 1, 2, 0, 2, 1, 2])
        numpy.testing.assert_allclose(known_yb, fjs.yb, rtol=RTOL)

        new_yb = fjs.update(self.new_data, k=2).yb
        known_new_yb = numpy.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1])
        numpy.testing.assert_allclose(known_new_yb, new_yb, rtol=RTOL)


class TestFindBin:
    def setup_method(self):
        self.V = load_example()

    def test_find_bin(self):
        toclass = [0, 1, 3, 5, 50, 70, 101, 202, 390, 505, 800, 5000, 5001]
        mc = FisherJenks(self.V, k=5)
        known = [0, 0, 0, 0, 0, 0, 1, 2, 3, 3, 4, 4, 4]
        numpy.testing.assert_array_equal(known, mc.find_bin(toclass))

        mc2 = FisherJenks(self.V, k=9)
        known = [0, 0, 0, 0, 2, 2, 3, 5, 7, 7, 8, 8, 8]
        numpy.testing.assert_array_equal(known, mc2.find_bin(toclass))


class TestMake:
    def setup_method(self):
        self.data = [
            numpy.linspace(-5, 5, num=5),
            numpy.linspace(-10, 10, num=5),
            numpy.linspace(-20, 20, num=5),
        ]
        self.ei = EqualInterval.make()
        self.q5r = Quantiles.make(k=5, rolling=True)

    def test_make(self):
        assert isinstance(self.ei, types.FunctionType)
        assert isinstance(self.q5r, types.FunctionType)

        assert hasattr(self.ei, "_options")
        assert self.ei._options == dict()
        assert hasattr(self.q5r, "_options")
        assert self.q5r._options == {"k": 5, "rolling": True}

    def test_apply(self):
        ei_classes = [self.ei(d) for d in self.data]
        known = [numpy.arange(0, 5, 1)] * 3
        numpy.testing.assert_allclose(known, ei_classes)

        q5r_classes = [self.q5r(d) for d in self.data]
        known = [[0, 1, 2, 3, 4], [0, 0, 2, 3, 4], [0, 0, 2, 4, 4]]
        accreted_data = set(self.q5r.__defaults__[0].y)
        all_data = set(numpy.asarray(self.data).flatten())
        assert accreted_data == all_data
        numpy.testing.assert_allclose(known, q5r_classes)


class TestBinC:
    def test_bin_c(self):
        bins = list(range(2, 8))
        y = numpy.array(
            [
                [7, 5, 6],
                [2, 3, 5],
                [7, 2, 2],
                [3, 6, 7],
                [6, 3, 4],
                [6, 7, 4],
                [6, 5, 6],
                [4, 6, 7],
                [4, 6, 3],
                [3, 2, 7],
            ]
        )

        expected = numpy.array(
            [
                [5, 3, 4],
                [0, 1, 3],
                [5, 0, 0],
                [1, 4, 5],
                [4, 1, 2],
                [4, 5, 2],
                [4, 3, 4],
                [2, 4, 5],
                [2, 4, 1],
                [1, 0, 5],
            ]
        )
        numpy.testing.assert_array_equal(expected, binC(y, bins))


class TestBin:
    def test_bin(self):
        y = numpy.array(
            [
                [7, 13, 14],
                [10, 11, 13],
                [7, 17, 2],
                [18, 3, 14],
                [9, 15, 8],
                [7, 13, 12],
                [16, 6, 11],
                [19, 2, 15],
                [11, 11, 9],
                [3, 2, 19],
            ]
        )
        bins = [10, 15, 20]
        expected = numpy.array(
            [
                [0, 1, 1],
                [0, 1, 1],
                [0, 2, 0],
                [2, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [2, 0, 1],
                [2, 0, 1],
                [1, 1, 0],
                [0, 0, 2],
            ]
        )

        numpy.testing.assert_array_equal(expected, bin(y, bins))


class TestBin1d:
    def test_bin1d(self):
        y = numpy.arange(100, dtype="float")
        bins = [25, 74, 100]
        binIds = numpy.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
            ]
        )
        counts = numpy.array([26, 49, 25])

        numpy.testing.assert_array_equal(binIds, bin1d(y, bins)[0])
        numpy.testing.assert_array_equal(counts, bin1d(y, bins)[1])


class TestNaturalBreaks:
    def setup_method(self):
        self.V = load_example()

    def test_natural_breaks(self):
        # assert expected, natural_breaks(values, k, itmax))
        assert True  # TODO: implement your test here

    def test_NaturalBreaks(self):
        nb = NaturalBreaks(self.V, 5)
        assert nb.k == 5
        assert len(nb.counts) == 5
        numpy.testing.assert_array_almost_equal(
            nb.counts, numpy.array([49, 3, 4, 1, 1])
        )

    def test_NaturalBreaks_stability(self):
        for i in range(10):
            nb = NaturalBreaks(self.V, 5)
            assert nb.k == 5
            assert len(nb.counts) == 5

    def test_NaturalBreaks_randomData(self):
        for i in range(10):
            V = numpy.random.random(50) * (i + 1)
            nb = NaturalBreaks(V, 5)
            assert nb.k == 5
            assert len(nb.counts) == 5


class TestHeadTailBreaks:
    def setup_method(self):
        x = list(range(1, 1000))
        y = []
        for i in x:
            y.append(i ** (-2))
        self.V = numpy.array(y)

    def test_HeadTailBreaks(self):
        htb = HeadTailBreaks(self.V)
        assert htb.k == 4
        assert len(htb.counts) == 4
        numpy.testing.assert_array_almost_equal(
            htb.counts, numpy.array([975, 21, 2, 1])
        )

    def test_HeadTailBreaks_doublemax(self):
        V = numpy.append(self.V, self.V.max())
        htb = HeadTailBreaks(V)
        assert htb.k == 4
        assert len(htb.counts) == 4
        numpy.testing.assert_array_almost_equal(
            htb.counts, numpy.array([980, 17, 1, 2])
        )

    def test_HeadTailBreaks_float(self):
        V = numpy.array([1 + 2**-52, 1, 1])
        htb = HeadTailBreaks(V)
        assert htb.k == 2
        assert len(htb.counts) == 2
        numpy.testing.assert_array_almost_equal(htb.counts, numpy.array([2, 1]))


class TestPrettyBreaks:
    def setup_method(self):
        self.V = load_example()

    def test_pretty(self):
        res = PrettyBreaks(self.V)
        assert res.k == 5
        numpy.testing.assert_array_equal(res.counts, [57, 0, 0, 0, 1])
        numpy.testing.assert_array_equal(res.bins, list(range(1000, 6000, 1000)))


class TestMapClassifier:
    def test_Map_Classifier(self):
        # map__classifier = Map_Classifier(y)
        assert True  # TODO: implement your test here

    def test___repr__(self):
        # map__classifier = Map_Classifier(y)
        # assert expected, map__classifier.__repr__())
        assert True  # TODO: implement your test here

    def test___str__(self):
        # map__classifier = Map_Classifier(y)
        # assert expected, map__classifier.__str__())
        assert True  # TODO: implement your test here

    def test_get_adcm(self):
        # map__classifier = Map_Classifier(y)
        # assert expected, map__classifier.get_adcm())
        assert True  # TODO: implement your test here

    def test_get_gadf(self):
        # map__classifier = Map_Classifier(y)
        # assert expected, map__classifier.get_gadf())
        assert True  # TODO: implement your test here

    def test_get_tss(self):
        # map__classifier = Map_Classifier(y)
        # assert expected, map__classifier.get_tss())
        assert True  # TODO: implement your test here


class TestEqualInterval:
    def setup_method(self):
        self.V = load_example()

    def test_EqualInterval(self):
        ei = EqualInterval(self.V)
        numpy.testing.assert_array_almost_equal(
            ei.counts, numpy.array([57, 0, 0, 0, 1])
        )
        numpy.testing.assert_array_almost_equal(
            ei.bins, numpy.array([822.394, 1644.658, 2466.922, 3289.186, 4111.45])
        )

        with pytest.raises(
            ValueError, match="Not enough unique values in array to form 5 classes."
        ):
            EqualInterval(numpy.array([1, 1, 1, 1]))


class TestPercentiles:
    def setup_method(self):
        self.V = load_example()

    def test_Percentiles(self):
        pc = Percentiles(self.V)
        numpy.testing.assert_array_almost_equal(
            pc.bins,
            numpy.array(
                [
                    1.35700000e-01,
                    5.53000000e-01,
                    9.36500000e00,
                    2.13914000e02,
                    2.17994800e03,
                    4.11145000e03,
                ]
            ),
        )
        numpy.testing.assert_array_almost_equal(
            pc.counts, numpy.array([1, 5, 23, 23, 5, 1])
        )


class TestBoxPlot:
    def setup_method(self):
        self.V = load_example()

    def test_BoxPlot(self):
        bp = BoxPlot(self.V)
        bins = numpy.array(
            [
                -5.28762500e01,
                2.56750000e00,
                9.36500000e00,
                3.95300000e01,
                9.49737500e01,
                4.11145000e03,
            ]
        )
        numpy.testing.assert_array_almost_equal(bp.bins, bins)


class TestQuantiles:
    def setup_method(self):
        self.V = load_example()

    def test_Quantiles(self):
        q = Quantiles(self.V, k=5)
        numpy.testing.assert_array_almost_equal(
            q.bins,
            numpy.array(
                [
                    1.46400000e00,
                    5.79800000e00,
                    1.32780000e01,
                    5.46160000e01,
                    4.11145000e03,
                ]
            ),
        )
        numpy.testing.assert_array_almost_equal(
            q.counts, numpy.array([12, 11, 12, 11, 12])
        )


class TestStdMean:
    def setup_method(self):
        self.V = load_example()

    def test_StdMean(self):
        std_mean = StdMean(self.V)
        numpy.testing.assert_array_almost_equal(
            std_mean.bins,
            numpy.array(
                [-967.36235382, -420.71712519, 672.57333208, 1219.21856072, 4111.45]
            ),
        )
        numpy.testing.assert_array_almost_equal(
            std_mean.counts, numpy.array([0, 0, 56, 1, 1])
        )


class TestMaximumBreaks:
    def setup_method(self):
        self.V = load_example()

    def test_MaximumBreaks(self):
        mb = MaximumBreaks(self.V, k=5)
        assert mb.k == 5
        numpy.testing.assert_array_almost_equal(
            mb.bins, numpy.array([146.005, 228.49, 546.675, 2417.15, 4111.45])
        )
        numpy.testing.assert_array_almost_equal(
            mb.counts, numpy.array([50, 2, 4, 1, 1])
        )

        with pytest.raises(
            ValueError, match="Not enough unique values in array to form 5 classes."
        ):
            MaximumBreaks(numpy.array([1, 1, 1, 1]))


class TestFisherJenks:
    def setup_method(self):
        self.V = load_example()

    def test_FisherJenks(self):
        fj = FisherJenks(self.V)
        assert fj.adcm == 799.24000000000001
        numpy.testing.assert_array_almost_equal(
            fj.bins, numpy.array([75.29, 192.05, 370.5, 722.85, 4111.45])
        )
        numpy.testing.assert_array_almost_equal(
            fj.counts, numpy.array([49, 3, 4, 1, 1])
        )


class TestJenksCaspall:
    def setup_method(self):
        self.V = load_example()

    def test_JenksCaspall(self):
        numpy.random.seed(10)
        jc = JenksCaspall(self.V, k=5)
        numpy.testing.assert_array_almost_equal(
            jc.counts, numpy.array([14, 13, 14, 10, 7])
        )
        numpy.testing.assert_array_almost_equal(
            jc.bins,
            numpy.array(
                [
                    1.81000000e00,
                    7.60000000e00,
                    2.98200000e01,
                    1.81270000e02,
                    4.11145000e03,
                ]
            ),
        )


class TestJenksCaspallSampled:
    def setup_method(self):
        self.V = load_example()

    def test_JenksCaspallSampled(self):
        numpy.random.seed(100)
        x = numpy.random.random(100000)
        jc = JenksCaspall(x)
        jcs = JenksCaspallSampled(x)
        numpy.testing.assert_array_almost_equal(
            jc.bins,
            numpy.array([0.19718393, 0.39655886, 0.59648522, 0.79780763, 0.99997979]),
        )
        numpy.testing.assert_array_almost_equal(
            jcs.bins,
            numpy.array([0.20856569, 0.41513931, 0.62457691, 0.82561423, 0.99997979]),
        )


class TestJenksCaspallForced:
    def setup_method(self):
        self.V = load_example()

    def test_JenksCaspallForced(self):
        numpy.random.seed(100)
        jcf = JenksCaspallForced(self.V, k=5)
        numpy.testing.assert_array_almost_equal(
            jcf.bins,
            numpy.array(
                [
                    1.34000000e00,
                    5.90000000e00,
                    1.67000000e01,
                    5.06500000e01,
                    4.11145000e03,
                ]
            ),
        )
        numpy.testing.assert_array_almost_equal(
            jcf.counts, numpy.array([12, 12, 13, 9, 12])
        )

        with pytest.raises(
            ValueError, match="Not enough unique values in array to form 5 classes."
        ):
            JenksCaspallForced(numpy.array([1, 1, 1, 1]))


class TestUserDefined:
    def setup_method(self):
        self.V = load_example()

    def test_UserDefined(self):
        bins = [20, max(self.V)]
        ud = UserDefined(self.V, bins)
        numpy.testing.assert_array_almost_equal(ud.bins, numpy.array([20.0, 4111.45]))
        numpy.testing.assert_array_almost_equal(ud.counts, numpy.array([37, 21]))

    def test_UserDefined_max(self):
        bins = numpy.array([20, 30])
        ud = UserDefined(self.V, bins)
        numpy.testing.assert_array_almost_equal(
            ud.bins, numpy.array([20.0, 30.0, 4111.45])
        )
        numpy.testing.assert_array_almost_equal(ud.counts, numpy.array([37, 4, 17]))

    def test_UserDefined_invariant(self):
        bins = [10, 20, 30, 40]
        ud = UserDefined(numpy.array([12, 12, 12]), bins)
        numpy.testing.assert_array_almost_equal(ud.bins, numpy.array([10, 20, 30, 40]))
        numpy.testing.assert_array_almost_equal(ud.counts, numpy.array([0, 3, 0, 0]))

    def test_UserDefined_lowest(self):
        bins = [20, max(self.V)]
        ud = UserDefined(self.V, bins, lowest=-1.0)
        numpy.testing.assert_array_almost_equal(ud.bins, numpy.array([20.0, 4111.45]))
        numpy.testing.assert_array_almost_equal(ud.counts, numpy.array([37, 21]))
        classes = ["[  -1.00,   20.00]", "(  20.00, 4111.45]"]
        assert ud.get_legend_classes() == classes


class TestStdMeanAnchor:
    def setup_method(self):
        self.V = load_example()

    def test_StdMeanAnchor(self):
        sm = StdMean(self.V, anchor=True)
        bins = numpy.array(
            [
                125.92810345,
                672.57333208,
                1219.21856072,
                1765.86378936,
                2312.50901799,
                2859.15424663,
                3405.79947527,
                3952.4447039,
                4111.45,
            ]
        )
        counts = numpy.array([50, 6, 1, 0, 0, 0, 0, 0, 1])
        numpy.testing.assert_array_almost_equal(sm.bins, bins)
        numpy.testing.assert_array_almost_equal(sm.counts, counts)


class TestMaxP:
    def setup_method(self):
        self.V = load_example()

    def test_MaxP(self):
        numpy.random.seed(100)
        mp = MaxP(self.V)
        numpy.testing.assert_array_almost_equal(
            mp.bins,
            numpy.array([3.16000e00, 1.26300e01, 1.67000e01, 2.04700e01, 4.11145e03]),
        )
        numpy.testing.assert_array_almost_equal(
            mp.counts, numpy.array([18, 16, 3, 1, 20])
        )

        with pytest.raises(
            ValueError, match="Not enough unique values in array to form 5 classes."
        ):
            MaxP(numpy.array([1, 1, 1, 1]))


class TestGadf:
    def setup_method(self):
        self.V = load_example()

    def test_gadf(self):
        qgadf = gadf(self.V)
        assert qgadf[0] == 15
        assert qgadf[-1] == 0.37402575909092828


class TestKClassifiers:
    def setup_method(self):
        self.V = load_example()

    def test_K_classifiers(self):
        numpy.random.seed(100)
        ks = KClassifiers(self.V)
        assert ks.best.name == "FisherJenks"
        assert ks.best.gadf == 0.84810327199081048
        assert ks.best.k == 4


class TestPooled:
    def setup_method(self):
        n = 20
        self.data = numpy.array([numpy.arange(n) + i * n for i in range(1, 4)]).T

    def test_pooled(self):
        res = Pooled(self.data, k=4)
        assert res.k == 4
        numpy.testing.assert_array_almost_equal(
            res.col_classifiers[0].counts, numpy.array([15, 5, 0, 0])
        )
        numpy.testing.assert_array_almost_equal(
            res.col_classifiers[-1].counts, numpy.array([0, 0, 5, 15])
        )
        numpy.testing.assert_array_almost_equal(
            res.global_classifier.counts, numpy.array([15, 15, 15, 15])
        )
        res = Pooled(self.data, classifier="BoxPlot", hinge=1.5)
        numpy.testing.assert_array_almost_equal(
            res.col_classifiers[0].bins, numpy.array([-9.5, 34.75, 49.5, 64.25, 108.5])
        )

    def test_pooled_bad_classifier(self):
        classifier = "Larry David"
        message = f"'{classifier}' not a valid classifier."
        with pytest.raises(ValueError, match=message):
            Pooled(self.data, classifier=classifier, k=4)


class TestPlots:
    def setup_method(self):
        n = 20
        self.data = numpy.array([numpy.arange(n) + i * n for i in range(1, 4)]).T

    @pytest.mark.mpl_image_compare
    def test_histogram_plot(self):
        ax = Quantiles(self.data).plot_histogram()
        return ax.get_figure()

    @pytest.mark.mpl_image_compare
    def test_histogram_plot_despine(self):
        ax = Quantiles(self.data).plot_histogram(despine=False)
        return ax.get_figure()

    @pytest.mark.mpl_image_compare
    def test_histogram_plot_linewidth(self):
        ax = Quantiles(self.data).plot_histogram(
            linewidth=3, linecolor="red", color="yellow"
        )
        return ax.get_figure()
