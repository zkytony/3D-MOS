#!/usr/bin/env python

from distutils.core import setup

setup(name='mos3d',
      version='1.0',
      description='Multi-object Search in 3D',
      install_requires=[
          'sciex',
          'numpy',
          'matplotlib',
          'pygame',        # for some tests
          'opencv-python',  # for some tests
          'scipy',
          'PyOpenGL',
          'PyOpenGL_accelerate'
      ],
      license="MIT",
      author='Kaiyu Zheng',
      author_email='kaiyutony@gmail.com',
      packages=['mos3d'])
