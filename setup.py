#!/usr/bin/env python

from distutils.core import setup

with open('README.md') as file:
    long_description = file.read()

setup(name='parmap',
      version='1.0.0',
      description=('A parallel version of map that can drop-in replace standard map ',
                   'in most cases, and harnessing multiple cores in a single machine. ',
                   'It is desinged to avoid unnecessary copy of large shared data structures '
                   'when doing parallel processing that does NOT modifies those data structures.'),
      long_description=long_description,
      author='Jianfu Chen',
      license='APACHE-2.0',
      url='https://github.com/fashandge/python-parmap',
      py_modules=['parmap'],
      classifiers=[
        'Development Status :: 1 - Beta',
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
)
