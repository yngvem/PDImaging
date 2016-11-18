# PDImaging
## A Python toolkit for variational and PDE based image processing
PDE based image processing is an important part of image processing, there are, however, no proper Python toolboxes for efficiently solving such problems. This toolbox is an attempt to fill that hole and uses a combination of Cython and C++ to speed things up. One big advantage of this toolbox is that all functions have references to where the respective algorithms were first published, as well as what source I've used for the algorithm to improve the ease at which scientists use this toolbox.

## Current project state
This is a project is written as a project on the side of my studies and are only a couple of weeks old, and therefore not anywhere near any state of completion. There only functions currently implemented are for improving image quality, denoising using Perona Malik iterations and ROF/TV-L2 denoising as well as TV-L2 regularised deconvolution.

### Immediate todo list:
1. Better understanding of distutils/setuptools, so that I won't need more than one setupfile, and for setuptools to be redundant.
2. Structure header files in own folder.
3. TVL1 deconvolution.
4. Improve deconvolution framework.
5. Start on segmentation algorithms.

### Required python packages:
I reccomend using an installation of the great Anaconda Python bundle, as that will have all required Python packages installed. If not, these are the packages this toolbox requires:

1. Numpy
2. Cython
3. Scikit-Image
4. Setuptools (I plan on making it possible to install with distutils in the future)

## About me
I am a graduate student on my first year of a Masters degree in image processing and computational biology at the Norwegian University of Life Sciences, with goals of writing a PhD within one of those fields. 

Feel free to contact me at yngve.m.moe@gmail.com.

