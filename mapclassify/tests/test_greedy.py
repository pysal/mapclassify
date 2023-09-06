import sys

import geopandas
import libpysal
import pytest

from ..greedy import greedy

PY39 = sys.version_info.major == 3 and sys.version_info.minor == 9

world = geopandas.read_file(
    "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
)
sw = libpysal.weights.Queen.from_dataframe(
    world, ids=world.index.to_list(), silence_warnings=True
)


@pytest.mark.filterwarnings("ignore:Geometry is in a geographic CRS.")
class TestGreedy:
    def test_default(self):
        colors = greedy(world)
        assert len(colors) == len(world)
        assert set(colors) == set([0, 1, 2, 3, 4])
        assert colors.value_counts().to_list() == [36, 36, 35, 35, 35]
        assert (colors.index == world.index).all()

    def test_rook(self):
        colors = greedy(world, sw="rook")
        assert len(colors) == len(world)
        assert set(colors) == set([0, 1, 2, 3, 4])
        assert colors.value_counts().to_list() == [36, 36, 35, 35, 35]

    def test_sw(self):
        colors = greedy(world, sw=sw)
        assert len(colors) == len(world)
        assert set(colors) == set([0, 1, 2, 3, 4])
        assert colors.value_counts().to_list() == [36, 36, 35, 35, 35]

    def test_min_distance(self):
        europe = world.loc[world.CONTINENT == "Europe"].to_crs(epsg=3035)
        colors = greedy(europe, min_distance=500000)
        assert len(colors) == len(europe)
        assert set(colors) == set(range(13))
        assert colors.value_counts().to_list() == [3] * 13

    def test_invalid_strategy(self):
        strategy = "spice melange"
        with pytest.raises(ValueError, match=f"'{strategy}' is not a valid strategy."):
            greedy(world, strategy=strategy)


@pytest.mark.filterwarnings("ignore:Geometry is in a geographic CRS.")
@pytest.mark.parametrize("pysal_geos", [None, 0])
class TestGreedyParams:
    def test_count(self, pysal_geos):
        colors = greedy(
            world, strategy="balanced", balance="count", min_distance=pysal_geos
        )
        assert len(colors) == len(world)
        assert set(colors) == set([0, 1, 2, 3, 4])
        assert colors.value_counts().to_list() == [36, 36, 35, 35, 35]

    def test_area(self, pysal_geos):
        colors = greedy(
            world, strategy="balanced", balance="area", min_distance=pysal_geos
        )
        assert len(colors) == len(world)
        assert set(colors) == set([0, 1, 2, 3, 4])
        assert colors.value_counts().to_list() == [55, 49, 39, 32, 2]

    def test_centroid(self, pysal_geos):
        colors = greedy(
            world, strategy="balanced", balance="centroid", min_distance=pysal_geos
        )
        assert len(colors) == len(world)
        assert set(colors) == set([0, 1, 2, 3, 4])
        assert colors.value_counts().to_list() == [39, 36, 36, 34, 32]

    def test_distance(self, pysal_geos):
        colors = greedy(
            world, strategy="balanced", balance="distance", min_distance=pysal_geos
        )
        assert len(colors) == len(world)
        assert set(colors) == set([0, 1, 2, 3, 4])
        assert colors.value_counts().to_list() == [38, 36, 35, 34, 34]

    def test_largest_first(self, pysal_geos):
        colors = greedy(world, strategy="largest_first", min_distance=pysal_geos)
        assert len(colors) == len(world)
        assert set(colors) == set([0, 1, 2, 3, 4])
        assert colors.value_counts().to_list() == [64, 49, 42, 21, 1]

    def test_random_sequential(self, pysal_geos):
        """based on random, no consistent results to be tested"""
        colors = greedy(world, strategy="random_sequential", min_distance=pysal_geos)
        assert len(colors) == len(world)

    def test_smallest_last(self, pysal_geos):
        colors = greedy(world, strategy="smallest_last", min_distance=pysal_geos)
        assert len(colors) == len(world)
        assert set(colors) == set([0, 1, 2, 3])
        # skip pn Python 3.9 due to networkx/networkx#3993
        if not PY39:
            assert colors.value_counts().to_list() == [71, 52, 39, 15]

    def test_independent_set(self, pysal_geos):
        colors = greedy(world, strategy="independent_set", min_distance=pysal_geos)
        assert len(colors) == len(world)
        assert set(colors) == set([0, 1, 2, 3, 4])
        assert colors.value_counts().to_list() == [91, 42, 26, 13, 5]

    def test_connected_sequential_bfs(self, pysal_geos):
        colors = greedy(
            world, strategy="connected_sequential_bfs", min_distance=pysal_geos
        )
        assert len(colors) == len(world)
        assert set(colors) == set([0, 1, 2, 3, 4])
        assert colors.value_counts().to_list() == [77, 46, 34, 18, 2]

    def test_connected_sequential_dfs(self, pysal_geos):
        colors = greedy(
            world, strategy="connected_sequential_dfs", min_distance=pysal_geos
        )
        assert len(colors) == len(world)
        assert set(colors) == set([0, 1, 2, 3, 4])
        assert colors.value_counts().to_list() == [75, 52, 34, 14, 2]

    def test_connected_sequential(self, pysal_geos):
        colors = greedy(world, strategy="connected_sequential", min_distance=pysal_geos)
        assert len(colors) == len(world)
        assert set(colors) == set([0, 1, 2, 3, 4])
        assert colors.value_counts().to_list() == [77, 46, 34, 18, 2]

    def test_saturation_largest_first(self, pysal_geos):
        colors = greedy(
            world, strategy="saturation_largest_first", min_distance=pysal_geos
        )
        assert len(colors) == len(world)
        assert set(colors) == set([0, 1, 2, 3])
        assert colors.value_counts().to_list() == [71, 47, 42, 17]

    def test_DSATUR(self, pysal_geos):
        colors = greedy(world, strategy="DSATUR", min_distance=pysal_geos)
        assert len(colors) == len(world)
        assert set(colors) == set([0, 1, 2, 3])
        assert colors.value_counts().to_list() == [71, 47, 42, 17]

    def test_index(self, pysal_geos):
        world["ten"] = world.index * 10
        reindexed = world.set_index("ten")
        colors = greedy(reindexed, min_distance=pysal_geos)
        assert len(colors) == len(world)
        assert set(colors) == set([0, 1, 2, 3, 4])
        assert colors.value_counts().to_list() == [36, 36, 35, 35, 35]
