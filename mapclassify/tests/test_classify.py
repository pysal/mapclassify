import geopandas
import libpysal
import pytest

import mapclassify


def _assertions(a, b):
    assert a.k == b.k
    assert (a.yb == b.yb).all()
    assert (a.bins == b.bins).all()
    assert (a.counts == b.counts).all()


class TestClassify:
    def setup_method(self):
        link_to_data = libpysal.examples.get_path("columbus.shp")
        gdf = geopandas.read_file(link_to_data)
        self.x = gdf["HOVAL"].values

    def test_box_plot(self):
        a = mapclassify.classify(self.x, "box_plot")
        b = mapclassify.BoxPlot(self.x)
        _assertions(a, b)

    def test_equal_interval(self):
        a = mapclassify.classify(self.x, "EqualInterval", k=3)
        b = mapclassify.EqualInterval(self.x, k=3)
        _assertions(a, b)

    def test_fisher_jenks(self):
        a = mapclassify.classify(self.x, "FisherJenks", k=3)
        b = mapclassify.FisherJenks(self.x, k=3)
        _assertions(a, b)

    # mapclassify\classifiers.py:FisherJenksSampled.__init__
    # 2028  ids = np.random.randint(0, n, int(n * pct)) 
    @pytest.mark.xfail(reason="Stochastic. Passing a.s. requires random samples "
                              "to be the same in both instances. ")
    def test_fisher_jenks_sampled(self):
        a = mapclassify.classify(
            self.x, "FisherJenksSampled", k=3, pct_sampled=0.5, truncate=False
        )
        b = mapclassify.FisherJenksSampled(self.x, k=3, pct=0.5, truncate=False)
        _assertions(a, b)

    def test_headtail_breaks(self):
        a = mapclassify.classify(self.x, "headtail_breaks")
        b = mapclassify.HeadTailBreaks(self.x)
        _assertions(a, b)

    def test_quantiles(self):
        a = mapclassify.classify(self.x, "quantiles", k=3)
        b = mapclassify.Quantiles(self.x, k=3)
        _assertions(a, b)

    def test_percentiles(self):
        a = mapclassify.classify(self.x, "percentiles", pct=[25, 50, 75, 100])
        b = mapclassify.Percentiles(self.x, pct=[25, 50, 75, 100])
        _assertions(a, b)

        a = mapclassify.classify(self.x, "prettybreaks")
        b = mapclassify.PrettyBreaks(self.x)
        _assertions(a, b)

    def test_jenks_caspall(self):
        a = mapclassify.classify(self.x, "JenksCaspall", k=3)
        b = mapclassify.JenksCaspall(self.x, k=3)
        _assertions(a, b)

    def test_jenks_caspall_forced(self):
        a = mapclassify.classify(self.x, "JenksCaspallForced", k=3)
        b = mapclassify.JenksCaspallForced(self.x, k=3)
        _assertions(a, b)


    # mapclassify\classifiers.py:JenksCaspallSampled.__init__
    # 2224  ids = np.random.randint(0, n, int(n * pct)) 
    @pytest.mark.xfail(reason="Stochastic. Passing a.s. requires random samples "
                              "to be the same in both instances. ")
    def test_jenks_caspall_sampled(self):
        a = mapclassify.classify(self.x, "JenksCaspallSampled", pct_sampled=0.5)
        b = mapclassify.JenksCaspallSampled(self.x, pct=0.5)
        _assertions(a, b)

    # KMeans iterates starting from a randomly generated centroids
    @pytest.mark.xfail(reason="Stochastic. Passing a.s. requires random centroids "
                              "to be the same in both instances. ")
    def test_natural_breaks(self):
        a = mapclassify.classify(self.x, "natural_breaks")
        b = mapclassify.NaturalBreaks(self.x)
        _assertions(a, b)

    # mapclassify\classifiers.py:MaxP._set_bins
    # 2656  rseeds = np.random.permutation(list(range(k))).tolist()
    # 2701  rseeds = np.random.permutation(list(range(k))).tolist()
    @pytest.mark.xfail(reason="Stochastic. Passing a.s. requires random selections "
                              "to be the same in both instances. ")
    def test_max_p_classifier(self):
        a = mapclassify.classify(self.x, "max_p", k=3, initial=50)
        b = mapclassify.MaxP(self.x, k=3, initial=50)
        _assertions(a, b)

    def test_std_mean(self):
        a = mapclassify.classify(self.x, "std_mean", multiples=[-1, -0.5, 0.5, 1])
        b = mapclassify.StdMean(self.x, multiples=[-1, -0.5, 0.5, 1])
        _assertions(a, b)

    def test_user_defined(self):
        a = mapclassify.classify(self.x, "user_defined", bins=[20, max(self.x)])
        b = mapclassify.UserDefined(self.x, bins=[20, max(self.x)])
        _assertions(a, b)

    def test_bad_classifier(self):
        classifier = "George_Costanza"
        with pytest.raises(ValueError, match="Invalid scheme: 'georgecostanza'"):
            mapclassify.classify(self.x, classifier)
