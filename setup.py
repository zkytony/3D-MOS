#!/usr/bin/env python

# Copyright 2022 Kaiyu Zheng
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from distutils.core import setup

setup(name='mos3d',
      version='1.0',
      description='Multi-object Search in 3D',
      install_requires=[
          'sciex',
          'numpy',
          'matplotlib',
          'pandas',
          'pygame',        # for some tests
          'opencv-python',  # for some tests
          'scipy==1.7.0',
          'PyOpenGL',
          'PyOpenGL_accelerate',
          'pomdp-py==1.2.4.5'
      ],
      license="MIT",
      author='Kaiyu Zheng',
      author_email='kaiyutony@gmail.com',
      packages=['mos3d'])
