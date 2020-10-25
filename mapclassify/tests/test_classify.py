from libpysal import examples

from mapclassify import classify

def test_classify():
    # data
    link_to_data = examples.get_path('columbus.shp')
    gdf = gpd.read_file(link_to_data)
    x = gdf['HOVAL'].values
    # quantiles
    mapclassify_bin(x, 'quantiles')
    mapclassify_bin(x, 'quantiles', k=3)
    # box_plot
    mapclassify_bin(x, 'box_plot')
    mapclassify_bin(x, 'box_plot', hinge=2)
    # headtail_breaks
    mapclassify_bin(x, 'headtail_breaks')   
    # percentiles
    mapclassify_bin(x, 'percentiles')
    mapclassify_bin(x, 'percentiles', pct=[25,50,75,100])
    # std_mean
    mapclassify_bin(x, 'std_mean')
    mapclassify_bin(x, 'std_mean', multiples=[-1,-0.5,0.5,1])
    # maximum_breaks
    mapclassify_bin(x, 'maximum_breaks')
    mapclassify_bin(x, 'maximum_breaks', k=3, mindiff=0.1)
    # natural_breaks, max_p_classifier
    mapclassify_bin(x, 'natural_breaks')
    mapclassify_bin(x, 'max_p_classifier', k=3, initial=50)
    # user_defined
    mapclassify_bin(x, 'user_defined', bins=[20, max(x)])