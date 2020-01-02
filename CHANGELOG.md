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

