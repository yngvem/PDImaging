# PDImaging
## A Python toolkit for variational and PDE based image processing
PDE based image processing is an important part of image processing, there are, however, no proper Python toolboxes for efficiently solving such problems. This toolbox is an attempt to fill that hole and uses a combination of Cython and C++ to speed things up. One big advantage of this toolbox is that all functions have references to where the respective algorithms were first published, as well as what source I've used for the algorithm to improve the ease at which scientists use this toolbox.

## Current project state
This is a project is written as a project on the side of my studies and are only a couple of weeks old, and therefore not anywhere near any state of completion. There only functions currently implemented are for improving image quality, denoising using Perona Malik iterations and ROF/TV-L2 denoising as well as TV-L2 regularised deconvolution.

### Immediate todo list:
1. Restructure the deconvolve cython file - there's reduntant code there
2. Improve setup files to make building with distutils possible
3. Combine setup files in one file if possible
4. TVL1 deconvolution.
5. Improve deconvolution framework.

### Long term todo list:
1. Change C++ code to use templates so image bitrate is conserved
2. Segmentation algorithms
3. Textures using wavelets
4. 3D-Implementations of some algorithms
5. Image inpainting (maybe)
6. Markov model techniques (Not variational or PDE based, but interesting nonetheless)

### Required python packages:
I reccomend using an installation of the great Anaconda Python bundle, as that will have all required Python packages installed. If not, these are the packages this toolbox requires:

1. Numpy
2. Cython
3. Scikit-Image
4. Scipy (for convolution)
5. Setuptools (I plan on making it possible to install with distutils in the future)

## About me
I am a graduate student on my first year of a Masters degree in image processing and computational biology at the Norwegian University of Life Sciences. I have completed a Bachelor's degree within applied mathematics and physics at the same university, with one year spent at the University of Manchester, where I wrote my undergraduate project. My project thesis is meant as an introduction to the mathematical background of tomographic reconstruction techniques, as well as an implementation and comparison of different algorithms. The MATLAB toolkit I made, as well as the thesis can be found at https://github.com/yngvem/CT_reconstruction 

## Contact:
Feel free to contact me, either about this project, or for general questions about tomography at yngve.m.moe@gmail.com.

