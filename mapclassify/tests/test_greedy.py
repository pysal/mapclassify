import geopandas as gpd
import pytest
from libpysal.weights import Queen

from ..greedy import greedy

world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
sw = Queen.from_dataframe(world, ids=world.index.to_list(), silence_warnings=True)


@pytest.mark.filterwarnings("ignore:Geometry is in a geographic CRS.")
def test_default():
    colors = greedy(world)
    assert len(colors) == len(world)
    assert set(colors) == set([0, 1, 2, 3, 4])
    assert colors.value_counts().to_list() == [36, 36, 35, 35, 35]
    assert (colors.index == world.index).all()


@pytest.mark.filterwarnings("ignore:Geometry is in a geographic CRS.")
@pytest.mark.parametrize("pysal_geos", [None, 0])
def test_count(pysal_geos):
    colors = greedy(
        world, strategy="balanced", balance="count", min_distance=pysal_geos
    )
    assert len(colors) == len(world)
    assert set(colors) == set([0, 1, 2, 3, 4])
    assert colors.value_counts().to_list() == [36, 36, 35, 35, 35]


@pytest.mark.filterwarnings("ignore:Geometry is in a geographic CRS.")
@pytest.mark.parametrize("pysal_geos", [None, 0])
def test_area(pysal_geos):
    colors = greedy(world, strategy="balanced", balance="area", min_distance=pysal_geos)
    assert len(colors) == len(world)
    assert set(colors) == set([0, 1, 2, 3, 4])
    assert colors.value_counts().to_list() == [55, 49, 39, 32, 2]


@pytest.mark.filterwarnings("ignore:Geometry is in a geographic CRS.")
@pytest.mark.parametrize("pysal_geos", [None, 0])
def test_centroid(pysal_geos):
    colors = greedy(
        world, strategy="balanced", balance="centroid", min_distance=pysal_geos
    )
    assert len(colors) == len(world)
    assert set(colors) == set([0, 1, 2, 3, 4])
    assert colors.value_counts().to_list() == [39, 36, 36, 34, 32]


@pytest.mark.filterwarnings("ignore:Geometry is in a geographic CRS.")
@pytest.mark.parametrize("pysal_geos", [None, 0])
def test_distance(pysal_geos):
    colors = greedy(
        world, strategy="balanced", balance="distance", min_distance=pysal_geos
    )
    assert len(colors) == len(world)
    assert set(colors) == set([0, 1, 2, 3, 4])
    assert colors.value_counts().to_list() == [38, 36, 35, 34, 34]


@pytest.mark.filterwarnings("ignore:Geometry is in a geographic CRS.")
@pytest.mark.parametrize("pysal_geos", [None, 0])
def test_largest_first(pysal_geos):
    colors = greedy(world, strategy="largest_first", min_distance=pysal_geos)
    assert len(colors) == len(world)
    assert set(colors) == set([0, 1, 2, 3, 4])
    assert colors.value_counts().to_list() == [64, 49, 42, 21, 1]


@pytest.mark.filterwarnings("ignore:Geometry is in a geographic CRS.")
@pytest.mark.parametrize("pysal_geos", [None, 0])
def test_random_sequential(pysal_geos):
    colors = greedy(world, strategy="random_sequential", min_distance=pysal_geos)
    assert len(colors) == len(world)
    # it is based on random, does not return consistent result to be tested


@pytest.mark.filterwarnings("ignore:Geometry is in a geographic CRS.")
@pytest.mark.parametrize("pysal_geos", [None, 0])
def test_smallest_last(pysal_geos):
    colors = greedy(world, strategy="smallest_last", min_distance=pysal_geos)
    assert len(colors) == len(world)
    assert set(colors) == set([0, 1, 2, 3])
    # assert colors.value_counts().to_list() == [71, 52, 39, 15]
    # skipped due to networkx/networkx#3993


@pytest.mark.filterwarnings("ignore:Geometry is in a geographic CRS.")
@pytest.mark.parametrize("pysal_geos", [None, 0])
def test_independent_set(pysal_geos):
    colors = greedy(world, strategy="independent_set", min_distance=pysal_geos)
    assert len(colors) == len(world)
    assert set(colors) == set([0, 1, 2, 3, 4])
    assert colors.value_counts().to_list() == [91, 42, 26, 13, 5]


@pytest.mark.filterwarnings("ignore:Geometry is in a geographic CRS.")
@pytest.mark.parametrize("pysal_geos", [None, 0])
def test_connected_sequential_bfs(pysal_geos):
    colors = greedy(world, strategy="connected_sequential_bfs", min_distance=pysal_geos)
    assert len(colors) == len(world)
    assert set(colors) == set([0, 1, 2, 3, 4])
    assert colors.value_counts().to_list() == [77, 46, 34, 18, 2]


@pytest.mark.filterwarnings("ignore:Geometry is in a geographic CRS.")
@pytest.mark.parametrize("pysal_geos", [None, 0])
def test_connected_sequential_dfs(pysal_geos):
    colors = greedy(world, strategy="connected_sequential_dfs", min_distance=pysal_geos)
    assert len(colors) == len(world)
    assert set(colors) == set([0, 1, 2, 3, 4])
    assert colors.value_counts().to_list() == [75, 52, 34, 14, 2]


@pytest.mark.parametrize("pysal_geos", [None, 0])
def test_connected_sequential(pysal_geos):
    colors = greedy(world, strategy="connected_sequential", min_distance=pysal_geos)
    assert len(colors) == len(world)
    assert set(colors) == set([0, 1, 2, 3, 4])
    assert colors.value_counts().to_list() == [77, 46, 34, 18, 2]


@pytest.mark.filterwarnings("ignore:Geometry is in a geographic CRS.")
@pytest.mark.parametrize("pysal_geos", [None, 0])
def test_saturation_largest_first(pysal_geos):
    colors = greedy(world, strategy="saturation_largest_first", min_distance=pysal_geos)
    assert len(colors) == len(world)
    assert set(colors) == set([0, 1, 2, 3])
    assert colors.value_counts().to_list() == [71, 47, 42, 17]


@pytest.mark.filterwarnings("ignore:Geometry is in a geographic CRS.")
@pytest.mark.parametrize("pysal_geos", [None, 0])
def test_DSATUR(pysal_geos):
    colors = greedy(world, strategy="DSATUR", min_distance=pysal_geos)
    assert len(colors) == len(world)
    assert set(colors) == set([0, 1, 2, 3])
    assert colors.value_counts().to_list() == [71, 47, 42, 17]


@pytest.mark.filterwarnings("ignore:Geometry is in a geographic CRS.")
def test_invalid_strategy():
    with pytest.raises(ValueError):
        greedy(world, strategy="invalid")


@pytest.mark.filterwarnings("ignore:Geometry is in a geographic CRS.")
def test_rook():
    colors = greedy(world, sw="rook")
    assert len(colors) == len(world)
    assert set(colors) == set([0, 1, 2, 3, 4])
    assert colors.value_counts().to_list() == [36, 36, 35, 35, 35]


@pytest.mark.filterwarnings("ignore:Geometry is in a geographic CRS.")
def test_sw():
    colors = greedy(world, sw=sw)
    assert len(colors) == len(world)
    assert set(colors) == set([0, 1, 2, 3, 4])
    assert colors.value_counts().to_list() == [36, 36, 35, 35, 35]


@pytest.mark.filterwarnings("ignore:Geometry is in a geographic CRS.")
@pytest.mark.parametrize("pysal_geos", [None, 0])
def test_index(pysal_geos):
    world["ten"] = world.index * 10
    reindexed = world.set_index("ten")
    colors = greedy(reindexed, min_distance=pysal_geos)
    assert len(colors) == len(world)
    assert set(colors) == set([0, 1, 2, 3, 4])
    assert colors.value_counts().to_list() == [36, 36, 35, 35, 35]


@pytest.mark.filterwarnings("ignore:Geometry is in a geographic CRS.")
def test_min_distance():
    europe = world.loc[world.continent == "Europe"].to_crs(epsg=3035)
    colors = greedy(europe, min_distance=500000)
    assert len(colors) == len(europe)
    assert set(colors) == set(range(13))
    assert colors.value_counts().to_list() == [3] * 13
