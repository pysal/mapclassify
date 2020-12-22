# Version 2.4.1 (2020-12-20)

This is a bug-fix release.

We closed a total of 9 issues (enhancements and bug fixes) through 3 pull requests, since our last release on 2020-12-13.

## Issues Closed
  - BUG: support series in sampled classifiers (#99)
  - BUG: FisherJenksSampled returns ValueError if Series is passed as y (#98)
  - REGR: fix invariant array regression (#101)
  - REGR: UserDefined classifier returns ValueError("Minimum and maximum of input data are equal, cannot create bins.") (#100)
  - [DOC] add example nb for new classify API (#91)
  - 2.4.0 Release (#97)

## Pull Requests
  - BUG: support series in sampled classifiers (#99)
  - REGR: fix invariant array regression (#101)
  - 2.4.0 Release (#97)

The following individuals contributed to this release:

  - Serge Rey
  - Martin Fleischmann
  - Stefanie Lumnitz
  
# Version 2.4.0 (2020-12-13)

We closed a total of 39 issues (enhancements and bug fixes) through 15 pull requests, since our last release on 2020-06-13.
Issues Closed

 - Remove timeout on tests. (#96)
 - BUG: HeadTailBreaks RecursionError due to floating point issue (#92)
 - Handle recursion error for head tails. (#95)
 - Add streamlined API (#72)
 - [API] add high-level API mapclassify.classify() (#90)
 - BUG: Fix mapclassify #88 (#89)
 - exclude Python 3.6 for Windows (#94)
 - CI: update conda action (#93)
 - EqualInterval unclear error when max_y - min_y = 0 (#88)
 - BUG: fix unordered series in greedy (#87)
 - BUG: greedy(strategy='balanced') does not return correct labels (#86)
 - Extra files in PyPI sdist (#56)
 - MAINT: fix repos name (#85)
 - DOC: content type for long description (#84)
 - MAINT: update gitcount notebook (#83)
 - Update documentations to include tutorial (#63)
 - build binder for notebooks (#71)
 - current version of mapclassify in docs? (#70)
 - 404 for notebook/tutorials links in docs (#79)
 - DOC: figs (#82)
 - DOCS: new images for tutorial (#81)
 - DOC: missing figs (#80)
 - DOCS: update documentation pages (#78)
 - Make networkx optional, remove xfail from greedy (#77)

## Pull Requests

 - Remove timeout on tests. (#96)
 - Handle recursion error for head tails. (#95)
 - [API] add high-level API mapclassify.classify() (#90)
 - BUG: Fix mapclassify #88 (#89)
 - exclude Python 3.6 for Windows (#94)
 - CI: update conda action (#93)
 - BUG: fix unordered series in greedy (#87)
 - MAINT: fix repos name (#85)
 - DOC: content type for long description (#84)
 - MAINT: update gitcount notebook (#83)
 - DOC: figs (#82)
 - DOCS: new images for tutorial (#81)
 - DOC: missing figs (#80)
 - DOCS: update documentation pages (#78)
 - Make networkx optional, remove xfail from greedy (#77)

The following individuals contributed to this release:

    Serge Rey
    Stefanie Lumnitz
    James Gaboardi
    Martin Fleischmann

 
# Version 2.3.0 (2020-06-13)
## Key Enhancements

- Topological coloring to ensure no two adjacent polygons share the same color.
- Pooled classification allows for the use of the same class intervals across maps.

## Details

We closed a total of 30 issues (enhancements and bug fixes) through 10 pull requests, since our last release on 2020-01-04.
## Issues Closed

  - Make networkx optional, remove xfail from greedy (#77)
  - BINDER: point to upstream (#76)
  - add binder badge (#75)
  - Binder (#74)
  - sys import missing from setup.py (#73)
  - [WIP] DOC: Updating tutorial (#66)
  - chorobrewer branch has begun (#27)
  - Is mapclassify code black? (#68)
  - Code format and README (#69)
  - Move testing over to github actions (#64)
  - Add pinning in pooled example documentation (#67)
  - Migrate to GHA (#65)
  - Add a Pooled classifier (#51)
  - Backwards compatability (#48)
  - Difference between Natural Breaks and Fisher Jenks schemes (#62)
  - ENH: add greedy (topological) coloring (#61)
  - Error while running mapclassify (#60)
  - Pooled (#59)
  - Invalid escape sequences in strings (#57)
  - 3.8, appveyor, deprecation fixes (#58)

## Pull Requests

  - Make networkx optional, remove xfail from greedy (#77)
  - BINDER: point to upstream (#76)
  - add binder badge (#75)
  - Binder (#74)
  - [WIP] DOC: Updating tutorial (#66)
  - Code format and README (#69)
  - Migrate to GHA (#65)
  - ENH: add greedy (topological) coloring (#61)
  - Pooled (#59)
  - 3.8, appveyor, deprecation fixes (#58)

## Acknowledgements

The following individuals contributed to this release:

  - Serge Rey
  - James Gaboardi
  - Eli Knaap
  - Martin Fleischmann

 
  
# Version 2.2.0 (2019-12-21)

This releases brings new functionality for [formatting of legend classes](https://github.com/sjsrey/geopandas/blob/legendkwds/examples/choro_legends.ipynb).

We closed a total of 21 issues (enhancements and bug fixes) through 9 pull requests, since our last release on 2019-06-28.

## Issues Closed
  - 2.2 (#54)
  - 2.2 (#53)
  - conda-forge UnsatisfiableError on windows and python 3.7 (#52)
  - [MAINT] updating supported Python versions in setup.py (#49)
  - BUG: RecursiveError in HeadTailBreaks (#46)
  - BUG: HeadTailBreaks raise RecursionError (#45)
  - BUG: UserDefined accepts only list if max not in bins (#47)
  - BUG: avoid deprecation warning in HeadTailBreaks (#44)
  - remove docs badge (#42)
  - Remove doc badge (#43)
  - Docs: moving to project pages on github and off rtd (#41)
  - BUG: Fix for downstream breakage in geopandas (#40)

## Pull Requests
  - 2.2 (#54)
  - 2.2 (#53)
  - [MAINT] updating supported Python versions in setup.py (#49)
  - BUG: RecursiveError in HeadTailBreaks (#46)
  - BUG: UserDefined accepts only list if max not in bins (#47)
  - BUG: avoid deprecation warning in HeadTailBreaks (#44)
  - Remove doc badge (#43)
  - Docs: moving to project pages on github and off rtd (#41)
  - BUG: Fix for downstream breakage in geopandas (#40)

The following individuals contributed to this release:

  - Serge Rey
  - James Gaboardi
  - Wei Kang
  - Martin Fleischmann


# Version 2.1.0 (2019-06-26)

We closed a total of 36 issues (enhancements and bug fixes) through 16 pull requests, since our last release on 2018-10-28.

## Issues Closed
  - ENH: dropping 3.5 support and adding 3.7 (#38)
  - ENH: plot method added to Mapclassify (#36)
  - ENH: keeping init keyword argument to avoid API breakage. (#35)
  - mapclassify.Natural_Break() does not return the specified k classes (#16)
  - Fix for #16 (#32)
  - Mixed usage of brewer2mpl and palettable.colorbrewer in color.py (#33)
  - Chorobrewer (#34)
  - conda-forge recipe needs some love (#14)
  - generating images for color selector (#31)
  - doc: bump version and dev setup docs (#30)
  - environment.yml (#29)
  - add color import and chorobrewer notebook (#28)
  - Chorobrewer (#26)
  - chorobrewer init (#25)
  - add badges for pypi, zenodo and docs (#24)
  - add geopandas and libpysal to test requirement (#23)
  - adjust changelog and delete tools/github_stats.py (#22)
  - add requirements_docs.txt to MANIFEST.in (#21)
  - gadf and K_classifiers not in __ini__.py (#18)
  - rel: 2.0.1 (#20)

## Pull Requests
  - ENH: dropping 3.5 support and adding 3.7 (#38)
  - ENH: plot method added to Mapclassify (#36)
  - ENH: keeping init keyword argument to avoid API breakage. (#35)
  - Fix for #16 (#32)
  - Chorobrewer (#34)
  - generating images for color selector (#31)
  - doc: bump version and dev setup docs (#30)
  - environment.yml (#29)
  - add color import and chorobrewer notebook (#28)
  - Chorobrewer (#26)
  - chorobrewer init (#25)
  - add badges for pypi, zenodo and docs (#24)
  - add geopandas and libpysal to test requirement (#23)
  - adjust changelog and delete tools/github_stats.py (#22)
  - add requirements_docs.txt to MANIFEST.in (#21)
  - rel: 2.0.1 (#20)

The following individuals contributed to this release:

  - Serge Rey
  - Wei Kang

# Version 2.0.1 (2018-10-28)

We closed a total of 12 issues (enhancements and bug fixes) through 5 pull requests, since our last release on 2018-08-10.

## Issues Closed
  - gadf and K_classifiers not in __ini__.py (#18)
  - rel: 2.0.1 (#20)
  - fix doctests (interactive examples in inline docstrings) (#19)
  - complete readthedocs configuration & add Slocum 2009 reference (#17)
  - prepping for a doc based release (#15)
  - new release on pypi (#10)
  - prepare for release 2.0.0 (#13)

## Pull Requests
  - rel: 2.0.1 (#20)
  - fix doctests (interactive examples in inline docstrings) (#19)
  - complete readthedocs configuration & add Slocum 2009 reference (#17)
  - prepping for a doc based release (#15)
  - prepare for release 2.0.0 (#13)

The following individuals contributed to this release:

  - Serge Rey
  - Wei Kang

# Version 2.0.0 (2018-08-10)

Starting from this release, mapclassify supports python 3+ only (currently 3.5
and 3.6).

This release also features a first stable version of mapclassify in
the process of pysal refactoring. There is a big change in the api in that we no
 longer provide an api module (`from mapclassify.api import Quantiles`). Instead,
 users will directly `from mapclassify import Quantiles`.

GitHub stats for 2017/08/18 - 2018/08/10

These lists are automatically generated, and may be incomplete or contain duplicates.

We closed a total of 8 issues, 4 pull requests and 4 regular issues;
this is the full list (generated with the script
:file:`tools/github_stats.py`):

Pull Requests (4):

* :ghpull:`12`: b'Clean up for next pypi release'
* :ghpull:`11`: b'move notebooks outside of the package'
* :ghpull:`9`: b'ENH: move classifiers up into init'
* :ghpull:`8`: b'Moving to python 3+'

Issues (4):

* :ghissue:`12`: b'Clean up for next pypi release'
* :ghissue:`11`: b'move notebooks outside of the package'
* :ghissue:`9`: b'ENH: move classifiers up into init'
* :ghissue:`8`: b'Moving to python 3+'


# Version 1.0.1 (2017-08-17)

- Warnings added when duplicate values make quantiles ill-defined
- Faster digitize in place of list comprehension
- Bug fix for consistent treatment of intervals (closed on the right, open on the left)

v<1.0.0dev> 2017-04-21

- alpha release

