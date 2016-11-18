# -*- coding: utf-8 -*-

"""
Compiler instructions for the Perona Malik anisotropic denoising algorithm
"""

__author__ = 'Yngve Mardal Moe'
__email__ = 'yngve.m.moe@gmail.com'

from setuptools import setup
from setuptools import Extension

from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "perona_malik",
        ["perona_malik.pyx"],
        extra_compile_args=['/openmp'],
        extra_link_args=['/openmp'],
        language="c++",
    )
]

setup(
    ext_modules = cythonize(ext_modules),
    include_dirs = [numpy.get_include()]
)
