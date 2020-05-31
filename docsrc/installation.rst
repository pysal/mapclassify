.. Installation

Installation
============

mapclassify supports python `3.6`_, `3.7`_, and `3.8`_. Please make sure that you are
operating in a python 3 environment.

Installing released version
---------------------------

mapclassify is available in on `conda`_ via the `conda-forge`_ channel::

  conda install -c conda-forge mapclassify


mapclassify is also available on the `Python Package Index`_. Therefore, you can either
install directly with `pip` from the command line::

  pip install -U mapclassify


or download the source distribution (.tar.gz) and decompress it to your selected
destination. Open a command shell and navigate to the decompressed folder.
Type::

  pip install .

Installing development version
------------------------------

Potentially, you might want to use the newest features in the development
version of mapclassify on github - `pysal/mapclassify`_ while have not been incorporated
in the Pypi released version. You can achieve that by installing `pysal/mapclassify`_
by running the following from a command shell::

  pip install git+https://github.com/pysal/mapclassify.git

You can  also `fork`_ the `pysal/mapclassify`_ repo and create a local clone of
your fork. By making changes
to your local clone and submitting a pull request to `pysal/mapclassify`_, you can
contribute to mapclassify development.

.. _3.6: https://docs.python.org/3.6/
.. _3.7: https://docs.python.org/3.7/
.. _3.8: https://docs.python.org/3.8/
.. _conda: https://docs.conda.io/en/latest/
.. _conda-forge: https://anaconda.org/conda-forge/mapclassify
.. _Python Package Index: https://pypi.org/project/mapclassify/
.. _pysal/mapclassify: https://github.com/pysal/mapclassify
.. _fork: https://help.github.com/articles/fork-a-repo/
