# -*- coding: utf-8 -*-

"""
Will write soon!
"""

__author__ = 'Yngve Mardal Moe'
__email__ = 'yngve.m.moe@gmail.com'

from easy_deconvolve import *
from perona_malik import *
from tv_fista import *

tv_fista_doc = """
Functions to perform denoising and deconvolutionusing the framework laid by
Rudin-Osher and Fatemi (ROF) in [1]. The minimization is performed using the
FISTA method derived by Amir Beck and Marc Teboulle in [2] as described in
their paper for TV-FISTA in [3]. The FISTA iterations have improved
convergence by utilizing the a momentum restart scheme, as proposed by
Brendan O'Donoghue, Emmanuel Candes in [4]. The denoising is performed using
paralellized C++ code that runs at a minimum of twice the speed of the
Scikit-Image Chambolle total variation solver.

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

perona_malik_doc = """
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

easy_deconvolve_doc = """
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