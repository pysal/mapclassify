import geopandas
import numpy as np
from numpy.testing import assert_array_equal
from mapclassify.util import get_color_array

world = geopandas.read_file(
    "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
).reset_index(drop=True)


def test_rgba():
    colors = get_color_array(world.area, cmap="viridis")[0]
    assert_array_equal(colors, np.array([68, 1, 84, 255])) 
    

def test_rgba_hex():
    colors = get_color_array(world.area, cmap="viridis", as_hex=True)[0]
    assert_array_equal(colors,'#440154') 
    
def test_rgba_nan():
    worldnan = world.copy()
    worldnan['area'] = worldnan.area
    worldnan.loc[0, 'area'] = np.nan
    colors = get_color_array(worldnan['area'], cmap="viridis", nan_color=[0,0,0,0])[0]
    assert_array_equal(colors, np.array([0, 0, 0, 0])) 

def test_rgba_nan_hex():
    worldnan = world.copy()
    worldnan['area'] = worldnan.area
    worldnan.loc[0, 'area'] = np.nan
    colors = get_color_array(worldnan['area'], cmap="viridis",nan_color=[0,0,0,0], as_hex=True)[0]
    assert_array_equal(colors, np.array(['#000000'])) 
    