import os
import sys
from distutils.command.build_py import build_py
from io import open
from os.path import join as pjoin
from os.path import relpath

from setuptools import find_packages, setup

import versioneer

package = "mapclassify"

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists("MANIFEST"):
    os.remove("MANIFEST")
curdir = os.path.abspath(os.path.dirname(__file__))

# This check resolves conda-forge build failures
# See the link below for original solution
# https://github.com/pydata/xarray/pull/2643/files#diff-2eeaed663bd0d25b7e608891384b7298R29-R30
needs_pytest = {"pytest", "test", "ptr"}.intersection(sys.argv)
setup_requires = ["pytest-runner"] if needs_pytest else []

with open("README.md", "r", encoding="utf8") as file:
    long_description = file.read()


def _get_requirements_from_files(groups_files):
    groups_reqlist = {}

    for k, v in groups_files.items():
        with open(v, "r") as f:
            pkg_list = f.read().splitlines()
        groups_reqlist[k] = pkg_list

    return groups_reqlist


def get_data_files():
    sep = os.path.sep
    # install the datasets
    data_files = {}
    root = pjoin(curdir, package, "datasets")
    for i in os.listdir(root):
        if i == "tests":
            continue
        path = pjoin(root, i)
        if os.path.isdir(path):
            data_files.update(
                {relpath(path, start=curdir).replace(sep, "."): ["*.csv", "*.dta"]}
            )

    return data_files


package_data = get_data_files()


def setup_package():
    _groups_files = {
        "base": "requirements.txt",
        "speedups": "requirements_speedups.txt",
        "tests": "requirements_tests.txt",
        "docs": "requirements_docs.txt",
    }

    reqs = _get_requirements_from_files(_groups_files)
    install_reqs = reqs.pop("base")
    extras_reqs = reqs

    setup(
        name=package,
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass({"build_py": build_py}),
        description="Classification Schemes for Choropleth Maps.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/pysal/%s" % package,
        maintainer="Serge Rey, Wei Kang",
        maintainer_email="sjsrey@gmail.com, weikang9009@gmail.com",
        setup_requires=setup_requires,
        tests_require=["pytest"],
        keywords="spatial statistics, geovisualizaiton",
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: GIS",
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
        ],
        license="3-Clause BSD",
        packages=find_packages(),
        py_modules=[package],
        package_data=package_data,
        install_requires=install_reqs,
        extras_require=extras_reqs,
        zip_safe=False,
    )


if __name__ == "__main__":
    setup_package()
