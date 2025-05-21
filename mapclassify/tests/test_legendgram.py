import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from libpysal import examples
from matplotlib.testing.decorators import image_comparison
from packaging.version import Version

from mapclassify import EqualInterval, Quantiles
from mapclassify.legendgram import _legendgram


class TestLegendgram:
    def setup_method(self):
        np.random.seed(42)
        self.data = np.random.normal(0, 1, 100)
        self.classifier = EqualInterval(self.data, k=5)

    def test_legendgram_returns_axis(self):
        """Test that _legendgram returns a matplotlib axis"""
        _, ax = plt.subplots(figsize=(8, 6))
        histax = _legendgram(self.classifier, ax=ax)
        plt.close()

        assert isinstance(histax, matplotlib.axes.Axes)

    def test_legendgram_standalone(self):
        """Test that _legendgram works without providing an axis"""
        histax = _legendgram(self.classifier)
        plt.close()

        assert isinstance(histax, matplotlib.axes.Axes)

    def test_legendgram_inset_false(self):
        """Test that _legendgram works with inset=False"""
        _, ax = plt.subplots(figsize=(8, 6))
        histax = _legendgram(self.classifier, ax=ax, inset=False)
        plt.close()

        # When inset=False, histax should be the same as ax
        assert histax is ax

    def test_legendgram_clip(self):
        """Test that _legendgram applies clip parameter correctly"""
        _, ax = plt.subplots(figsize=(8, 6))
        clip_range = (-2, 2)
        histax = _legendgram(self.classifier, ax=ax, clip=clip_range)
        xlim = histax.get_xlim()
        plt.close()

        assert xlim[0] == clip_range[0]
        assert xlim[1] == clip_range[1]

    def test_legendgram_tick_params(self):
        """Test that _legendgram applies tick_params correctly"""
        _, ax = plt.subplots(figsize=(8, 6))
        custom_tick_params = {"labelsize": 20, "rotation": 45}
        _ = _legendgram(self.classifier, ax=ax, tick_params=custom_tick_params)
        plt.close()

    def test_legendgram_frameon(self):
        """Test that _legendgram applies frameon parameter correctly"""
        _, ax = plt.subplots(figsize=(8, 6))
        histax = _legendgram(self.classifier, ax=ax, frameon=True)
        is_frame_on = histax.get_frame_on()
        plt.close()

        assert is_frame_on

    @pytest.mark.skipif(
        Version(matplotlib.__version__) >= Version("3.11.0.dev"),
        reason="change of font rendering breaks image comparison",
        # once 3.11 lands, we should update expected and test against that
    )
    @image_comparison(
        baseline_images=["legendgram_default"],
        extensions=["png"],
        remove_text=False,
        tol=0.05,
    )
    def test_legendgram_default(self):
        """Test default legendgram appearance"""
        _, ax = plt.subplots(figsize=(8, 6))
        _legendgram(self.classifier, ax=ax)

    @pytest.mark.skipif(
        Version(matplotlib.__version__) >= Version("3.11.0.dev"),
        reason="change of font rendering breaks image comparison",
        # once 3.11 lands, we should update expected and test against that
    )
    @image_comparison(
        baseline_images=["legendgram_vlines"],
        extensions=["png"],
        remove_text=False,
        tol=0.05,
    )
    def test_legendgram_vlines(self):
        """Test legendgram with vertical lines"""
        _, ax = plt.subplots(figsize=(8, 6))
        _legendgram(self.classifier, ax=ax, vlines=True, vlinecolor="red", vlinewidth=2)

    @pytest.mark.skipif(
        Version(matplotlib.__version__) >= Version("3.11.0.dev"),
        reason="change of font rendering breaks image comparison",
        # once 3.11 lands, we should update expected and test against that
    )
    @image_comparison(
        baseline_images=["legendgram_cmap"],
        extensions=["png"],
        remove_text=False,
        tol=0.05,
    )
    def test_legendgram_cmap(self):
        """Test legendgram with custom colormap"""
        _, ax = plt.subplots(figsize=(8, 6))
        _legendgram(self.classifier, ax=ax, cmap="plasma")

    @pytest.mark.skipif(
        Version(matplotlib.__version__) >= Version("3.11.0.dev"),
        reason="change of font rendering breaks image comparison",
        # once 3.11 lands, we should update expected and test against that
    )
    @image_comparison(
        baseline_images=["legendgram_position"],
        extensions=["png"],
        remove_text=False,
        tol=0.05,
    )
    def test_legendgram_position(self):
        """Test legendgram with custom position"""
        _, ax = plt.subplots(figsize=(8, 6))
        _legendgram(
            self.classifier, ax=ax, loc="upper right", legend_size=("40%", "30%")
        )

    @pytest.mark.skipif(
        Version(matplotlib.__version__) >= Version("3.11.0.dev"),
        reason="change of font rendering breaks image comparison",
        # once 3.11 lands, we should update expected and test against that
    )
    @image_comparison(
        baseline_images=["legendgram_map"],
        extensions=["png"],
        remove_text=False,
        tol=0.05,
    )
    def test_legendgram_map(self):
        """Test with geopandas map"""
        data = gpd.read_file(examples.get_path("south.shp")).to_crs(epsg=5070)
        ax = data.plot("DV80", k=10, scheme="Quantiles")
        classifier = Quantiles(data["DV80"].values, k=10)
        classifier.plot_legendgram(
            ax=ax, legend_size=("50%", "20%"), loc="upper left", clip=(2, 10)
        )
        ax.set_axis_off()
