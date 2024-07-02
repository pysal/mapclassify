import geopandas
import numpy as np
from mapclassify.util import get_rgba

world = geopandas.read_file(
    "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
)


def test_rgba():
    colors = get_rgba(world.area, cmap="viridis")[0]
    assert colors == [
        np.float64(68.08602),
        np.float64(1.24287),
        np.float64(84.000825),
        np.float64(255.0),
    ]
