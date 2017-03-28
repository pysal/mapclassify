from setuptools import setup

try:
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:
    from distutils.command.build_py import build_py

setup(name='mapclassify',
      version='1.0.0',
      description="""Classification schemes for choropleth maps.""",
      url= 'https://github.com/sjsrey/mapclassify', #github repo
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
      packages=[],
      install_requires=['numpy', 'scipy', 'pandas',],
      zip_safe=False,
      cmdclass = {'build.py':build_py})
