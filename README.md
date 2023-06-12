# mapclassify: Classification Schemes for Choropleth Maps

[![Continuous Integration](https://github.com/pysal/mapclassify/actions/workflows/testing.yml/badge.svg)](https://github.com/pysal/mapclassify/actions/workflows/testing.yml)
[![codecov](https://codecov.io/gh/pysal/mapclassify/branch/main/graph/badge.svg)](https://codecov.io/gh/pysal/mapclassify)
[![PyPI version](https://badge.fury.io/py/mapclassify.svg)](https://badge.fury.io/py/mapclassify)
[![DOI](https://zenodo.org/badge/88918063.svg)](https://zenodo.org/badge/latestdoi/88918063)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pysal/mapclassify/main)

`mapclassify` implements a family of classification schemes for choropleth maps.
Its focus is on the determination of the number of classes, and the assignment
of observations to those classes. It is intended for use with upstream mapping
and geovisualization packages (see
[geopandas](https://geopandas.org/mapping.html) and
[geoplot](https://residentmario.github.io/geoplot/user_guide/Customizing_Plots.html))
that handle the rendering of the maps.

For further theoretical background see [Rey, S.J., D. Arribas-Bel, and L.J. Wolf (2020) "Geographic Data Science with PySAL and the PyData Stack”](https://geographicdata.science/book/notebooks/05_choropleth.html).

## Using `mapclassify`
Load built-in example data reporting employment density in 58 California counties:

```python
>>> import mapclassify
>>> y = mapclassify.load_example()
>>> y.mean()
125.92810344827588
>>> y.min(), y.max()
(0.13, 4111.4499999999998)

```

## Map Classifiers Supported

### BoxPlot

```python
>>> mapclassify.BoxPlot(y)
BoxPlot

     Interval        Count
--------------------------
(   -inf,  -52.88] |     0
( -52.88,    2.57] |    15
(   2.57,    9.36] |    14
(   9.36,   39.53] |    14
(  39.53,   94.97] |     6
(  94.97, 4111.45] |     9
```



### EqualInterval

```python
>>> mapclassify.EqualInterval(y)
EqualInterval

     Interval        Count
--------------------------
[   0.13,  822.39] |    57
( 822.39, 1644.66] |     0
(1644.66, 2466.92] |     0
(2466.92, 3289.19] |     0
(3289.19, 4111.45] |     1
```

### FisherJenks

```python
>>> import numpy as np
>>> np.random.seed(123456)
>>> mapclassify.FisherJenks(y, k=5)
FisherJenks

     Interval        Count
--------------------------
[   0.13,   75.29] |    49
(  75.29,  192.05] |     3
( 192.05,  370.50] |     4
( 370.50,  722.85] |     1
( 722.85, 4111.45] |     1
```

### FisherJenksSampled

```python
>>> np.random.seed(123456)
>>> x = np.random.exponential(size=(10000,))
>>> mapclassify.FisherJenks(x, k=5)
FisherJenks

   Interval      Count
----------------------
[ 0.00,  0.64] |  4694
( 0.64,  1.45] |  2922
( 1.45,  2.53] |  1584
( 2.53,  4.14] |   636
( 4.14, 10.61] |   164

>>> mapclassify.FisherJenksSampled(x, k=5)
FisherJenksSampled

   Interval      Count
----------------------
[ 0.00,  0.70] |  5020
( 0.70,  1.63] |  2952
( 1.63,  2.88] |  1454
( 2.88,  5.32] |   522
( 5.32, 10.61] |    52
```

### HeadTailBreaks

```python
>>> mapclassify.HeadTailBreaks(y)
HeadTailBreaks

     Interval        Count
--------------------------
[   0.13,  125.93] |    50
( 125.93,  811.26] |     7
( 811.26, 4111.45] |     1
```

### JenksCaspall

```python
>>> mapclassify.JenksCaspall(y, k=5)
JenksCaspall

     Interval        Count
--------------------------
[   0.13,    1.81] |    14
(   1.81,    7.60] |    13
(   7.60,   29.82] |    14
(  29.82,  181.27] |    10
( 181.27, 4111.45] |     7
```

### JenksCaspallForced

```python
>>> mapclassify.JenksCaspallForced(y, k=5)
JenksCaspallForced

     Interval        Count
--------------------------
[   0.13,    1.34] |    12
(   1.34,    5.90] |    12
(   5.90,   16.70] |    13
(  16.70,   50.65] |     9
(  50.65, 4111.45] |    12
```

### JenksCaspallSampled

```python
>>> mapclassify.JenksCaspallSampled(y, k=5)
JenksCaspallSampled

     Interval        Count
--------------------------
[   0.13,   12.02] |    33
(  12.02,   29.82] |     8
(  29.82,   75.29] |     8
(  75.29,  192.05] |     3
( 192.05, 4111.45] |     6
```

### MaxP

```python
>>> mapclassify.MaxP(y)
MaxP

     Interval        Count
--------------------------
[   0.13,    8.70] |    29
(   8.70,   16.70] |     8
(  16.70,   20.47] |     1
(  20.47,   66.26] |    10
(  66.26, 4111.45] |    10
```

### [MaximumBreaks](notebooks/maximum_breaks.ipynb)

```python
>>> mapclassify.MaximumBreaks(y, k=5)
MaximumBreaks

     Interval        Count
--------------------------
[   0.13,  146.00] |    50
( 146.00,  228.49] |     2
( 228.49,  546.67] |     4
( 546.67, 2417.15] |     1
(2417.15, 4111.45] |     1
```

### NaturalBreaks

```python
>>> mapclassify.NaturalBreaks(y, k=5)
NaturalBreaks

     Interval        Count
--------------------------
[   0.13,   75.29] |    49
(  75.29,  192.05] |     3
( 192.05,  370.50] |     4
( 370.50,  722.85] |     1
( 722.85, 4111.45] |     1
```

### Quantiles

```python
>>> mapclassify.Quantiles(y, k=5)
Quantiles

     Interval        Count
--------------------------
[   0.13,    1.46] |    12
(   1.46,    5.80] |    11
(   5.80,   13.28] |    12
(  13.28,   54.62] |    11
(  54.62, 4111.45] |    12
```

### Percentiles

```python
>>> mapclassify.Percentiles(y, pct=[33, 66, 100])
Percentiles

     Interval        Count
--------------------------
[   0.13,    3.36] |    19
(   3.36,   22.86] |    19
(  22.86, 4111.45] |    20
```

### PrettyBreaks
```python
>>> np.random.seed(123456)
>>> x = np.random.randint(0, 10000, (100,1))
>>> mapclassify.PrettyBreaks(x)
Pretty

      Interval         Count
----------------------------
[  300.00,  2000.00] |    23
( 2000.00,  4000.00] |    15
( 4000.00,  6000.00] |    18
( 6000.00,  8000.00] |    24
( 8000.00, 10000.00] |    20
 ```

### StdMean

```python
>>> mapclassify.StdMean(y)
StdMean

     Interval        Count
--------------------------
(   -inf, -967.36] |     0
(-967.36, -420.72] |     0
(-420.72,  672.57] |    56
( 672.57, 1219.22] |     1
(1219.22, 4111.45] |     1
```
### UserDefined

```python
>>> mapclassify.UserDefined(y, bins=[22, 674, 4112])
UserDefined

     Interval        Count
--------------------------
[   0.13,   22.00] |    38
(  22.00,  674.00] |    18
( 674.00, 4112.00] |     2
```

## Alternative API 

As of version 2.4.0 the API has been extended. A `classify` function is now
available for a streamlined interface:

```python
>>> classify(y, 'boxplot')                                  
BoxPlot                   

     Interval        Count
--------------------------
(   -inf,  -52.88] |     0
( -52.88,    2.57] |    15
(   2.57,    9.36] |    14
(   9.36,   39.53] |    14
(  39.53,   94.97] |     6
(  94.97, 4111.45] |     9

```




## Use Cases

### Creating and using a classification instance

```python
>>> bp = mapclassify.BoxPlot(y)
>>> bp
BoxPlot

     Interval        Count
--------------------------
(   -inf,  -52.88] |     0
( -52.88,    2.57] |    15
(   2.57,    9.36] |    14
(   9.36,   39.53] |    14
(  39.53,   94.97] |     6
(  94.97, 4111.45] |     9

>>> bp.bins
array([ -5.28762500e+01,   2.56750000e+00,   9.36500000e+00,
         3.95300000e+01,   9.49737500e+01,   4.11145000e+03])
>>> bp.counts
array([ 0, 15, 14, 14,  6,  9])
>>> bp.yb
array([5, 1, 2, 3, 2, 1, 5, 1, 3, 3, 1, 2, 2, 1, 2, 2, 2, 1, 5, 2, 4, 1, 2,
       2, 1, 1, 3, 3, 3, 5, 3, 1, 3, 5, 2, 3, 5, 5, 4, 3, 5, 3, 5, 4, 2, 1,
       1, 4, 4, 3, 3, 1, 1, 2, 1, 4, 3, 2])

```

### Binning new data

```python
>>> bp = mapclassify.BoxPlot(y)
>>> bp
BoxPlot

     Interval        Count
--------------------------
(   -inf,  -52.88] |     0
( -52.88,    2.57] |    15
(   2.57,    9.36] |    14
(   9.36,   39.53] |    14
(  39.53,   94.97] |     6
(  94.97, 4111.45] |     9
>>> bp.find_bin([0, 7, 3000, 48])
array([1, 2, 5, 4])

```
Note that `find_bin` does not recalibrate the classifier:
```python
>>> bp
BoxPlot

     Interval        Count
--------------------------
(   -inf,  -52.88] |     0
( -52.88,    2.57] |    15
(   2.57,    9.36] |    14
(   9.36,   39.53] |    14
(  39.53,   94.97] |     6
(  94.97, 4111.45] |     9
```
### Apply

```python
>>> import mapclassify 
>>> import pandas
>>> from numpy import linspace as lsp
>>> data = [lsp(3,8,num=10), lsp(10, 0, num=10), lsp(-5, 15, num=10)]
>>> data = pandas.DataFrame(data).T
>>> data
          0          1          2
0  3.000000  10.000000  -5.000000
1  3.555556   8.888889  -2.777778
2  4.111111   7.777778  -0.555556
3  4.666667   6.666667   1.666667
4  5.222222   5.555556   3.888889
5  5.777778   4.444444   6.111111
6  6.333333   3.333333   8.333333
7  6.888889   2.222222  10.555556
8  7.444444   1.111111  12.777778
9  8.000000   0.000000  15.000000
>>> data.apply(mapclassify.Quantiles.make(rolling=True))
   0  1  2
0  0  4  0
1  0  4  0
2  1  4  0
3  1  3  0
4  2  2  1
5  2  1  2
6  3  0  4
7  3  0  4
8  4  0  4
9  4  0  4

```


## Development Notes

Because we use `geopandas` in development, and geopandas has stable `mapclassify` as a dependency, setting up a local development installation involves creating a conda environment, then replacing the stable `mapclassify` with the development version of `mapclassify` in the development environment. This can be accomplished with the following steps:


```
conda-env create -f environment.yml
conda activate mapclassify
conda remove -n mapclassify mapclassify
pip install -e .
```

