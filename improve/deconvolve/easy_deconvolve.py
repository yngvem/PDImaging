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

__author__ = 'Yngve Mardal Moe'
__email__ = 'yngve.m.moe@gmail.com'

import numpy as np
import scipy.signal as sig
from skimage.filters import gaussian
from tv_deconvolve import deconvolve_fista


def deconvolve_tv(image, psf, noise_level=0.05, min_value=0, max_value=1,
                  intermediate_it=30, it=40, intermediate_eps=1e-3, eps=1e-5):
    """Computes the total variation regularized deconvolution of a given image,
    with the point spread function psf. This is computed using the FISTA
    method [2] and the framework derived in [3].

    :param image: Image to deconvolve.
    :param psf: Point spread function to invert.
    :param noise_level: Regularization parameter, higher means noisier data.
    :param min_value: Minimum pixel intensity.
    :param max_value: Maximum pixel intensity.
        :param intermediate_it: Iterations per proximal gradient computation.
    :param it: No. of FISTA iterations.
    :param intermediate_eps: Convergence level of proximal gradient computation.
    :param eps: Convergence level deconvolution iterations.
    :returns: Deconvoluted image.
    """
    psf_adjoint = np.rot90(psf, 2)
    filter = lambda x: sig.convolve_2d(x, psf, mode='same')
    adjoint_filter = lambda x: sig.convolve_2d(x, psf_adjoint, mode='same')
    return deconvolve_fista(image, filter, adjoint_filter, noise_level,
                            it=it, intermediate_it=intermediate_it, eps=eps,
                            intermediate_eps=intermediate_eps,
                            min_value=min_value, max_value=max_value,
                            lipschitz=find_lipschitz(image,
                                           lambda x: adjoint_filter(filter(x))))


def easy_gaussian_denoise(image, std, noise_level=0.05, min_value=0,
                          max_value=1, intermediate_it=30, it=40,
                          intermediate_eps=1e-3, eps=1e-5):
    """Performs total variation regularized deconvolution of image with a
    gaussian blurring kernel that has given standard deviation. This is
    performed using the FISTA method [2] and the framework derived in [3] as
    well as a restarting scheme as described in [4].

    :param image: Image to deblur.
    :param std: Standard deviation (radius) of the gaussian blurring kernel
    to invert.
    :param noise_level: Regularization parameter - Almost allways less than 0.
    :param min_value: Minimum pixel value.
    :param max_value: Maximum pixel value.
    :param intermediate_it: Iterations per proximal gradient computation.
    :param it: No. of FISTA iterations.
    :param intermediate_eps: Convergence level of proximal gradient computation.
    :param eps: Convergence level deconvolution iterations.
    :return: Deblurred image
    """
    filter = lambda x: gaussian(x, std)
    return deconvolve_fista(image, filter, filter, noise_level,
                            it=it, intermediate_it=intermediate_it, eps=eps,
                            intermediate_eps=intermediate_eps,
                            min_value=min_value, max_value=max_value,
                            lipschitz=find_lipschitz(image,
                                           lambda x: filter(filter(x))))


def find_lipschitz(x0, operator):
    """Use power iterations to find the Lipschitz constant of a linear
    operator O: V -> V. To find the Lipschitz constant for the gradient of:
        ||Ax - b||,
    with linear operator A, constant vector b and variable x, one has to
    compute the Lipschitz constant of the operator [A'A].

    :param x0: Initial vector x0.
    :param operator: Function that corresponds to the operator, takes one
    vector as argument and returns a vector of the same size.
    :returns: Lipschitz constant of the operator, returns 1.1 if the
    Lipschitz constant is less than 1.1 (for stability reasons)."""
    x0 = np.copy(x0)
    lip = np.linalg.norm(x0)
    for i in range(20):
        x0 /= lip
        x0 = operator(x0)
        lip = np.linalg.norm(x0)
    return lip if lip > 1.1 else 1.1
