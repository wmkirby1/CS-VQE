'''
Created on 22 Oct 2014

@author: andrew
'''
from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'oneZOnState',
  ext_modules = cythonize("oneZOnState.pyx"),
)