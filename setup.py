# -*- coding: utf-8 -*-

__author__ = 'Yngve Mardal Moe'
__email__ = 'yngve.m.moe@gmail.com'

from setuptools import setup
from Cython.Build import cythonize
import numpy
import platform

if platform.system() == 'Windows':
	args = ['/openmp']
else:
	args = ['-fopenmp']

setup(
    name='PDImaging',
    version='0.1',
    description='Python toolkit for PDE based image processing',
    author='Yngve Mardal Moe',
    author_email='yngve.m.moe@gmail.com',
    url='https://github.com/yngvem/PDImaging',
    ext_modules=cythonize(
    		'improve/*.pyx',
        	extra_compile_args=args,
        	extra_link_args=args,
        	language='c++',
        	compiler_directives={'embedsignature': True}
		),
    include_dirs=[numpy.get_include()]
)
