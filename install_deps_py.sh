#!/bin/bash

echo "Updating conda....."
/opt/conda/bin/conda update --yes 

echo "Installing conda mapclassify stack....."

conda config --set always_yes yes --set changeps1 no --set show_channel_urls true
conda config --add channels conda-forge --force
conda config --set channel_priority strict
conda install --quiet --yes \
  'scikit-learn' \
  'seaborn' \
  'libpysal=4.2.0' \
  'rtree'

#conda install --channel conda-forge geopandas

echo "Installing doc dependencies...."
pip install -U  sphinx sphinx_gallery sphinxcontrib-bibtex sphinx_bootstrap_theme sphinxcontrib-napoleon

echo "Installing testing dependencies...."

conda install -c conda-forge -c defaults --quiet --yes \
      'nose' \
      'nose-progressive' \
      'nose-exclude' \
      'coverage' \
      'coveralls' \
      'pytest' \
      'pytest-cov' \
      'pytest-mpl'

conda install -c conda-forge  --quiet --yes descartes
