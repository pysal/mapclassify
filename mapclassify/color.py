"""
color handling for mapping and geovisualization
"""

from palettable import colorbrewer
from collections import defaultdict

k_maps = {}
for ctype in ['Sequential', 'Qualitative', 'Diverging']:
    k_maps[ctype] = defaultdict(list)
    cmaps = colorbrewer.COLOR_MAPS[ctype]
    for cmap in cmaps:
        kvalues = list(cmaps[cmap].keys())
        for k in kvalues:
            k = int(k)
            k_maps[ctype][k].append(cmap)

sequential = colorbrewer.COLOR_MAPS['Sequential']
diverging = colorbrewer.COLOR_MAPS['Diverging']
qualitative = colorbrewer.COLOR_MAPS['Qualitative']



IMG_DIR = "img"

def get_cmaps_type(ctype='Sequential'):
    """Helper access colormaps for a ctype"""
    ctype = colorbrewer.COLOR_MAPS[ctype]
    return ctype


def make_color_bar_images(directory=IMG_DIR, width=1.0, height=0.2):
    """
    Create color bar images for use in selection drop-downs

    Arguments
    ---------

    directory: string
              directory to store images

    width: float
           inches for the width of the color ramp image

    height: float
           inches for the height of the color ramp image


    Notes
    -----
    Creates one image for each color map in the colorbrewer schemes from palettable.
    Each image will be named `cmap_k.png` where cmap is the name of cmap from palettable, and k is the number of classes
    """
    for ctype_key in ['Diverging', 'Sequential', 'Qualitative']:
        ctype = colorbrewer.COLOR_MAPS[ctype_key]
        for cmap_key, cmap in ctype.items():
            for k, cmap_k in cmap.items():
                cmap = colorbrewer.get_map(cmap_key, ctype_key, int(k))
                fname = "{dir}/{cmap_key}_{k}.png".format(dir=directory, cmap_key=cmap_key, k=k)
                cmap.save_discrete_image(filename=fname,size=(width, height))


def load_color_bar_image(cmap, k, directory=IMG_DIR):
    """
    Load image for a color bar

    Arguments
    ---------
    cmap: string
          palettable cmap string

    k: int
       number of classes


    directory: string
              directory to store images

    """

    fname = "{dir}/{cmap}_{k}.png".format(dir=directory, cmap=cmap, k=k)
    return fname


def get_hex_colors(cmap, ctype, k):
    """return list of hex colors for cmap

    Arguments
    ---------

    cmap: string
          Blues, PrGn,......RdBu

    ctype: string
           Sequential, Diverging, Qualitative

    k: int
       number of classes/colors

    Returns
    -------
    list hex codes for k colors

    Example
    -------
    >>> get_hex_colors('Blues', 'sequential', 5)
    ['#EFF3FF', '#BDD7E7', '#6BAED6', '#3182BD', '#08519C']

    >>> get_hex_colors('sequential', 'Blues', 5)
    Cmap not defined: sequential Blues 5

    """
    try:
        return colorbrewer.get_map(cmap, ctype, k).hex_colors
    except:
        print('Cmap not defined:', cmap, ctype, k)


