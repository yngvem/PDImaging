# -*- coding: utf-8 -*-

"""
Functions to perform denoising using the framework laid by Rudin-Osher and
Fatemi (ROF) in [1]. The minimization is performed using the FISTA method
derived by Amir Beck and Marc Teboulle in [2] as described in their paper for
TV-FISTA in [3]. The FISTA iterations have improved convergence by utilizing
the a momentum restart scheme, as proposed by Brendan O'Donoghue, Emmanuel
Candes in [4]. The denoising is performed using C code that runs at a minimum
of twice the speed of the Scikit-Image Chambolle total variation solver.

[1] Rudin, L, et al. "Nonlinear total variation based noise removal algorithms"
    Physica D: Nonlinear Phenomena (1992)
[2] Beck, A and Teboulle, M. "A fast iterative shrinkage-thresholding
    algorithm for linear inverse problems."
    SIAM journal on imaging sciences 2.1 (2009).
[3] Beck, A & Teboulle, M. "Fast Gradient-Based Algorithms for Constrained
    Total Variation Image Denoising and Deblurring Problems."
    IEEE Transactions on Image Processing (2009).
[4] O'Donoghue, B & Candes, E. "Adaptive Restart for Accelerated Gradient
    Schemes." Foundations of Computational Mathematics (2012)
"""

__author__ = "Yngve Mardal Moe"
__email__ = "yngve.m.moe@gmail.com"


import numpy as np
cimport numpy as np

cdef extern from "tv_fista.h":
    void TV_FISTA(double* image, double* raw, double gamma,
                  double min_intensity, double max_intensity, int max_it,
                  double eps, int y, int x)


cdef void c_tv_fista(np.ndarray[double, ndim=2, mode='c'] image,
                  np.ndarray[double, ndim=2, mode='c'] raw,
                  double gamma, double min_intensity, double max_intensity,
                  int it, double eps):
    y, x = np.shape(image)
    TV_FISTA(<double *> image.data, <double *> raw.data, gamma, min_intensity,
         max_intensity, it, eps, y, x)



cpdef tv_fista(raw, gamma, it = 100, eps=1e-4,
             min_intensity = 0,  max_intensity = 1):
    """Performs TV-denoising using the ROF model [1] with the FISTA method as as
    described in [3].
    :raw: Image to denoise.
    :gamma: Regularization constant/noise level.
    :it: Maximum number of iterations.
    :eps: Convergence level
    :min_intensity: Minimum image value
    :max_intensity: Maximum image value
    """
    image = np.zeros(np.shape(raw))
    c_tv_fista(image, raw, gamma, min_intensity, max_intensity, it, eps)
    return image
