import geopandas
import numpy as np
from numpy.testing import assert_array_equal
from mapclassify.util import get_rgba

world = geopandas.read_file(
    "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
)


def test_rgba():
    colors = get_rgba(world.area, cmap="viridis")[0]
    assert_array_equal(colors, np.array([68, 1, 84, 255])) 
    
