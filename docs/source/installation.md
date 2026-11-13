# Installation

mapclassify supports Python 3.12 and later. Please make sure that you are
operating in a Python 3 environment.

## Installing released version

mapclassify is available on [conda] via the [conda-forge] channel:

```bash
conda install -c conda-forge mapclassify
```

mapclassify is also available on the [Python Package Index]. Therefore, you can either
install directly with `pip` from the command line:

```bash
pip install -U mapclassify
```

or download the source distribution (.tar.gz) and decompress it to your selected
destination. Open a command shell and navigate to the decompressed folder.
Type:

```bash
pip install .
```

## Installing development version

Potentially, you might want to use the newest features in the development
version of mapclassify on github - [pysal/mapclassify] while have not been incorporated
in the Pypi released version. You can achieve that by installing [pysal/mapclassify]
by running the following from a command shell:

```bash
pip install git+https://github.com/pysal/mapclassify.git
```

You can  also [fork] the [pysal/mapclassify] repo and create a local clone of
your fork. By making changes
to your local clone and submitting a pull request to [pysal/mapclassify], you can
contribute to mapclassify development.

[conda]: https://docs.conda.io/en/latest/
[conda-forge]: https://anaconda.org/conda-forge/mapclassify
[Python Package Index]: https://pypi.org/project/mapclassify/
[pysal/mapclassify]: https://github.com/pysal/mapclassify
[fork]: https://help.github.com/articles/fork-a-repo/
