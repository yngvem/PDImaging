# PDImaging
## This was a fun side project I had last year, however, it might have been somewhat too ambitious. I might come back to this project later once I am finished with my masters degree.

## A Python toolkit for variational and PDE based image processing
This toolbox is an attempt to fill that hole and uses a combination of Cython and C++ to speed things up. One big advantage of this toolbox is that all functions have references to where the respective algorithms were first published, as well as what source I've used for the algorithm to improve the ease at which scientists use this toolbox.

## Current project state
This project is written on the side of my studies and are only, thus updates will be sporadic and it is not near any stat of completion yet. There only functions currently implemented are for improving image quality, denoising using Perona Malik iterations and ROF/TV-L2 denoising as well as TV-L2 regularised deconvolution.

### Immediate todo list:
1. Improve setup files to make building with distutils possible.
2. TVL1 deconvolution.
3. Create Guided Poisson solvers.
4. Create TGV denoising and deconvolution framework.
5. Improve deconvolution framework.

### Long term todo list:
1. Change C++ code to use templates so image bitrate is conserved
2. Segmentation algorithms
3. 3D-Implementations of some algorithms
4. Color implementations of algorithms
5. Image inpainting (maybe)

### Required python packages:
I reccomend using an installation of the great Anaconda Python bundle, as that will have all required Python packages installed. If not, these are the packages this toolbox requires:

1. Numpy
2. Cython
3. Scikit-Image
4. Scipy (for convolution)
