import geopandas
import libpysal
import matplotlib
import matplotlib.pyplot as plt
import pytest
from matplotlib.testing.decorators import image_comparison

from mapclassify import shift_colormap, truncate_colormap, vba_choropleth


class TestValueByAlphaChoropleth:
    def setup_method(self):
        self.gdf = geopandas.read_file(libpysal.examples.get_path("columbus.shp"))
        self.x = "HOVAL"
        self.y = "CRIME"

    @image_comparison(["no_classify_default"], **pytest.image_comp_kws)
    def test_no_classify_default(self):
        fig, ax = vba_choropleth(self.x, self.y, self.gdf)

        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)

    @image_comparison(["pass_in_ax"], **pytest.image_comp_kws)
    def test_pass_in_ax(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig, ax = vba_choropleth(self.x, self.y, self.gdf, ax=ax)

        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)

    @image_comparison(["classify_xy_redblue"], **pytest.image_comp_kws)
    def test_classify_xy_redblue(self):
        fig, ax = vba_choropleth(
            self.x,
            self.y,
            self.gdf,
            x_classification_kwds={"classifier": "quantiles"},
            y_classification_kwds={"classifier": "quantiles"},
            cmap="RdBu",
        )

        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)

    @image_comparison(["divergent_revert_alpha_min_alpha"], **pytest.image_comp_kws)
    def test_divergent_revert_alpha_min_alpha(self):
        fig, ax = vba_choropleth(
            self.x,
            self.y,
            self.gdf,
            x_classification_kwds={"classifier": "fisher_jenks", "k": 3},
            y_classification_kwds={"classifier": "fisher_jenks", "k": 3},
            cmap="berlin",
            divergent=True,
            revert_alpha=True,
            min_alpha=0.5,
        )

        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)

    @image_comparison(["userdefined_colors"], **pytest.image_comp_kws)
    def test_userdefined_colors(self):
        color_list = ["#a1dab4", "#41b6c4", "#225ea8"]
        fig, ax = vba_choropleth(self.x, self.y, self.gdf, cmap=color_list)

        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)

    @image_comparison(["shifted_colormap"], **pytest.image_comp_kws)
    def test_shifted_colormap(self):
        mid08 = shift_colormap("RdBu", midpoint=0.8, name="RdBu_08")

        assert mid08.name == "RdBu_08"

        fig, ax = vba_choropleth(self.x, self.y, self.gdf, cmap=mid08)

        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)

    @image_comparison(["truncated_colormap"], **pytest.image_comp_kws)
    def test_truncated_colormap(self):
        trunc0206 = truncate_colormap("RdBu", minval=0.2, maxval=0.6, n=2)

        assert trunc0206.name == "trunc(RdBu,0.20,0.60)"

        fig, ax = vba_choropleth(self.x, self.y, self.gdf, cmap=trunc0206)

        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_legend_error(self):
        with pytest.raises(
            ValueError,
            match=(
                "Plotting a legend requires classification for both the `x` and `y` "
                "variables. See `x_classification_kwds` and `y_classification_kwds`."
            ),
        ):
            vba_choropleth(self.x, self.y, self.gdf, legend=True)

    @image_comparison(["legend"], **pytest.image_comp_kws_legend_text)
    def test_legend(self):
        fig, ax = vba_choropleth(
            self.x,
            self.y,
            self.gdf,
            x_classification_kwds={"classifier": "quantiles"},
            y_classification_kwds={"classifier": "quantiles"},
            legend=True,
        )

        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)

    @image_comparison(["legend_kwargs"], **pytest.image_comp_kws_legend_text)
    def test_legend_kwargs(self):
        fig, ax = vba_choropleth(
            self.x,
            self.y,
            self.gdf,
            x_classification_kwds={"classifier": "quantiles"},
            y_classification_kwds={"classifier": "quantiles"},
            legend=True,
            legend_kwargs={"x_label": self.x, "y_label": self.y},
        )

        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)
