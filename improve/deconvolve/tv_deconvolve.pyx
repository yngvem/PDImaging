# -*- coding: utf-8 -*-

"""
Functions to perform deconvolution using the framework laid by Rudin-Osher and
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


cdef extern from "../../header_files/tv_fista.h":
    void TV_FISTA(double* image, double* raw, double gamma,
                  double min_value, double max_value, int max_it,
                  double eps, int y, int x)
    double TV_deconvolve_problem(double* image, double*decon_image, double*raw,
                                 double gamma, int y, int x)


cdef double c_TV_deconvolve_problem(np.ndarray[double, ndim=2, mode='c'] image,
                                  np.ndarray[double, ndim=2, mode='c'] decon_im,
                                  np.ndarray[double, ndim=2, mode='c'] raw,
                                  double gamma, int y, int x):
    return TV_deconvolve_problem(<double *> image.data,<double *> decon_im.data,
                                 <double *>  raw.data, gamma, y, x)

cdef TV_decon_problem(decon_im, raw, gamma, filter):
    """Evaluates the TV deconvolution problem function
    :param decon_im: Deconvoluted image
    :param raw: (Noisy) image to deconvolve
    :param gamma: Regularization constant/noise level
    :param filter: Function performing the convolution filter we want to remove
    """
    image = filter(decon_im)
    y, x = np.shape(image)
    return c_TV_deconvolve_problem(image, decon_im, raw, gamma, y, x)


cdef void c_tv_fista(np.ndarray[double, ndim=2, mode='c'] image,
                  np.ndarray[double, ndim=2, mode='c'] raw,
                  double gamma, double min_value, double max_value,
                  int it, double eps):
    y, x = np.shape(image)
    TV_FISTA(<double *> image.data, <double *> raw.data, gamma, min_value,
         max_value, it, eps, y, x)



cdef tv_fista(raw, gamma, it = 100, eps=1e-4,
             min_value = 0,  max_value = 1):
    """Performs TV-denoising using the ROF model [1] with the FISTA method as as
    described in [3].
    :param raw: Image to denoise.
    :param gamma: Regularization constant/noise level.
    :param it: Maximum number of iterations.
    :param eps: Convergence level
    :param min_value: Minimum image value
    :param max_value: Maximum image value
    """
    image = np.zeros(np.shape(raw))
    c_tv_fista(image, raw, gamma, min_value, max_value, it, eps)
    return image


cpdef deconvolve_fista(raw, filter, adjoint_filter, gamma, it=50,
                       intermediate_it=30, lipschitz=1.3, eps=1e-4,
                       intermediate_eps=1e-4, min_value=0, max_value=1):

    """Performs TV-deconvolution using the ROF model [1] using the FISTA
    method as described in [3] and a restart scheme as described in [4].
    :param raw: Image do deconvolve
    :param filter: Function, takes image as input and returns the
    image transformed by the convolution operator we want to invert
    remove, denoted F
    :param adjoint_filter: Same as filter, but with the adjoint convolution
                           operator instead (convolution using the same point
                           spread function as :filter: only rotated 180*),
                           denoted F'
    :param gamma: Regularization constant - noise level
    :param it: Number of FISTA iterations
    :param intermediate_it: Number of FISTA-denoise iterations per iteration
    :param lipschitz: Maximum bound for the Lipschitz constant of the:
                      [F'*F]
                      Operator, where * denotes composition.
                      This can be found as the sum of all the values in the
                      point spread function corresponding to F'*F (equals 1
                      for most blurring operators).
    :param eps: Convergence limit for the FISTA iterations
    :param intermediate_eps: Convergence limit for the denoise iterations
    :param min_value: The minimum image value
    :param max_value: The maximum image value
    """
    # Initialize
    cdef np.ndarray[double, ndim=2, mode='c'] image = np.zeros(np.shape(raw))
    cdef np.ndarray[double, ndim=2, mode='c'] prev_it = np.copy(image)
    cdef np.ndarray[double, ndim=2, mode='c'] momentum = np.copy(prev_it)
    cdef double t = 1
    cdef double t_1 = 1
    cdef double fn = TV_decon_problem(image, raw, gamma, filter)
    cdef double fn_1

    # FISTA iterations
    cdef int i
    for i in range(it):
        prev_it = np.copy(image)
        fn_1 = fn

        image = tv_fista(
            momentum - (2/lipschitz)*adjoint_filter(filter(momentum) - raw),
            2*gamma/lipschitz, intermediate_it, intermediate_eps,
            min_value, max_value)
        fn = TV_decon_problem(image, raw, gamma, filter)
        # Test wrt. the restart scheme:
        if fn_1 < fn:
            t = 1
            t_1 = 1
            image = np.copy(prev_it)
        # Test convergence
        elif fn_1 - fn < eps*fn:
            break
        else:
            t_1 = t
            t = (1 + np.sqrt(1 + 4*t*t))/2
            momentum = image + ((t_1-1)/t)*(image - prev_it)

    return image