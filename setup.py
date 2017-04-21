from setuptools import setup, find_packages
import os
from os.path import relpath, join as pjoin

try:
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:
    from distutils.command.build_py import build_py


curdir = os.path.abspath(os.path.dirname(__file__))

packages = find_packages()

def get_data_files():
    sep = os.path.sep
    # install the datasets
    data_files = {}
    root = pjoin(curdir, "mapclassify", "datasets")
    for i in os.listdir(root):
        if i is "tests":
            continue
        path = pjoin(root, i)
        if os.path.isdir(path):
            data_files.update({relpath(path, start=curdir).replace(sep, ".") : ["*.csv",
                                                                  "*.dta"]})
    # add all the tests and results files
    for r, ds, fs in os.walk(pjoin(curdir, "statsmodels")):
        r_ = relpath(r, start=curdir)
        if r_.endswith('results'):
            data_files.update({r_.replace(sep, ".") : ["*.csv",
                                                       "*.txt",
                                                       "*.dta"]})

    return data_files


package_data = get_data_files()

setup(name='mapclassify',
      version='1.0.0dev0',
      description="""Classification schemes for choropleth maps.""",
      url= 'https://github.com/pysal/mapclassify',
      maintainer="Serge Rey",
      maintainer_email="sjsrey@gmail.com",
      test_suite = 'nose.collector',
      tests_require=['nose'],
      keywords='spatial statistics, geovisualizaiton',
      classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: GIS',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.5',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4'
        ],
      license='3-Clause BSD',
      packages=packages,
      package_data = package_data,
      install_requires=['numpy', 'scipy', 'pandas',],
      zip_safe=False,
      cmdclass = {'build.py':build_py})
