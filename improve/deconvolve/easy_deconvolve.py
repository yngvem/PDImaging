# -*- coding: utf-8 -*-

"""

"""

__author__ = 'Yngve Mardal Moe'
__email__ = 'yngve.m.moe@gmail.com'

import numpy as np
import scipy.signal as sig
from tv_deconvolve import deconvolve_fista


def easy_deconvolve(image, psf, noise_level=0.1, min_value=0, max_value=1):
    """Computes the deconvoluted image using a total variation
    regularization, with noise_level denotes how much the image is smoothed
    and is usually below 1.

    :param image: Image to deconvolve.
    :param psf: Point spread function to invert.
    :param noise_level: Regularization parameter, higher means noisier data.
    :param min_value: Minimum pixel intensity.
    :param max_value: Maximum pixel intensity.
    :returns: Deconvoluted image


    Technnical details:
    TV-deconvolution using the ROF model [1] computed with the FISTA
    method as described in [3] and a restart scheme as described in [4].
    """
    psf_adjoint = np.rot90(psf, 2)
    filter = lambda x: sig.convolve_2d(x, psf, mode='same')
    adjoint_filter = lambda x: sig.convolve_2d(x, psf_adjoint, mode='same')
    return deconvolve_fista(image, filter, adjoint_filter, noise_level,
                            min_value, max_value)

