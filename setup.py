#!/usr/bin/env python

#from distutils.core import setup
from codecs import open
import re
import os
from os import path
from setuptools import setup, find_packages

def read(*names, **kwargs):
    with open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='parmap',
      version=find_version('parmap/__init__.py'),
      description=('A parallel version of map that can drop-in replace standard map ',
                   'in most cases. ',
                   'It is desinged to avoid unnecessary copy of large shared data structures.'
                  ),
      long_description=long_description,
      author='Jianfu Chen',
      license='APACHE-2.0',
      url='https://github.com/fashandge/parmap',
      #py_modules=['parmap'],
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: Linux',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
      ],
      # You can just specify the packages manually here if your project is
      # simple. Or you can use find_packages().
      packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
      # If there are data files included in your packages that need to be
      # installed, specify them here.  If using Python 2.6 or less, then these
      # have to be included in MANIFEST.in as well.
      #package_data={
      #    'parmap': ['README.md'],
      #},
      # include README.md to . dir
      data_files=[('', ['README.md'])],
      include_package_data=True,
      test_suite='nose.collector',
      tests_require=['nose', 'nose-cover3'],
      zip_safe=False,
      )
