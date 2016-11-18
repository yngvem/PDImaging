# -*- coding: utf-8 -*-

__author__ = 'Yngve Mardal Moe'
__email__ = 'yngve.m.moe@gmail.com'

from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
import numpy
import platform

if platform.system() == 'Windows':
    ext_modules = [
        Extension(
            'tv_deconvolve',
            ['tv_deconvolve.pyx'],
            extra_compile_args=['/openmp'],
            extra_link_args=['/openmp'],
            language='c++',
        ),
    ]
else:
    ext_modules = [
        Extension(
            'tv_deconvolve',
            ['tv_deconvolve.pyx'],
            extra_compile_args=['-fopenmp'],
            extra_link_args=['-fopenmp'],
            language='c++',
        ),
    ]

setup(
    name='PDImaging',
    version='0.01',
    description='Python toolkit for PDE based image processing',
    author='Yngve Mardal Moe',
    author_email='yngve.m.moe@gmail.com',
    url='https://github.com/yngvem/PDImaging',
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()]
)
