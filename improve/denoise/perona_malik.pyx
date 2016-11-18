# -*- coding: utf-8 -*-

"""
Functions to perform denoising using the anisotropic denoising framework laid by
Perona and Malik in [1], as described in the book "Geometric Partial
Differential Equations for Image Processing" by Guillermo Sapiro [2]. These
functions include the standard gradient descent implementation described in
the book as well as the (obvious) accelerated extension version,
using the accelerated gradient descent algorithm Nesterov proposed in [3]
(ref. from [4]) as described in [4]. It also includes automatic edge level
detection using robust statistics interpretation of Perona-Malik diffusion
from [5], using MAD outlier detection from both of which are described in [2].

This implementation uses C++ code with speedups from OpenMP parallel loops.

[1] Perona, P. and Malik, J. "Scale-space and edge detection using
    anisotropic diffusion" IEEE Transactions on Pattern Analysis and
    Machine Intelligence (1990).
[2] Sapiro, G. "Geometric Partial Differential Equations for Image Processing"
    Camebridge University Press 2001.
[3] Y. E. Nesterov, "A method for solving the convex programming problem with
    convergence rate O(1/k^2)" Dokl. Akad. Nauk SSSR, 269 (1983) (in Russian).
[4] Beck, A and Teboulle, M. "A fast iterative shrinkage-thresholding
    algorithm for linear inverse problems."
    SIAM journal on imaging sciences 2.1 (2009).
[5] You, Y. L. et al. "Behavioral analysis of anisotropic diffusion in image
    processing" IEEE Trans. Image Process. (1996)
"""

__author__ = "Yngve Mardal Moe"
__email__ = "yngve.m.moe@gmail.com"


import numpy as np
cimport numpy as np

cdef extern from "../../header_files/perona_malik.h":
    void ext_perona_malik(double* image, double* raw, double edge_level,
                          double step_length, int method, int max_it,
                          int y, int x)
    void ext_fast_perona_malik(double* image, double* raw, double edge_level,
                               double step_length, int method, int max_it,
                               int y, int x)


# Wrapper for the C++ implementation of the Perona-Malik denoising algorithm
cdef c_perona_malik(np.ndarray[double, ndim=2, mode='c'] image,
                    np.ndarray[double, ndim=2, mode='c'] raw,
                    double edge_level, double step_length, int method,
                    int max_it, int y, int x):
    ext_perona_malik(<double *> image.data, <double *> raw.data,
                     edge_level, step_length, method, max_it, y, x)


# Python function for the Perona-Malik denoising algorithm
cpdef perona_malik(raw,  method='lorenz', edge_level = 'auto',
                   step_length = 0.5, it = 20):
    y, x = np.shape(raw)
    image = np.zeros(np.shape(raw))

    if method == 'tukey':
        method = 2
    else:
        method = 1
    if edge_level == 'auto':
        edge_level = -1

    c_perona_malik(image, raw, edge_level, step_length, method, it, y, x)
    return image


# Wrapper for the C++ implementation of the Nesterov accelerated Perona-Malik
# denoising algorithm
cdef c_fast_perona_malik(np.ndarray[double, ndim=2, mode='c'] image,
                        np.ndarray[double, ndim=2, mode='c'] raw,
                        double edge_level, double step_length, int method,
                        int max_it, int y, int x):
    ext_fast_perona_malik(<double *> image.data, <double *> raw.data,
                          edge_level, step_length, method, max_it, y, x)


# Python function for the Nesterov accelerated Perona-Malik denoising algorithm
cpdef fast_perona_malik(raw, method='lorenz',
                        edge_level = 'auto', step_length = 1, it = 20):
    y, x = np.shape(raw)
    image = np.zeros(np.shape(raw))

    if method == 'tukey' or method == 1:
        method = 2
    else:
        method = 1
    if edge_level == 'auto':
        edge_level = -1

    c_fast_perona_malik(image, raw, edge_level, step_length, method, it, y, x)
    return image