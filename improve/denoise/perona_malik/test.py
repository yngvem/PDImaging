# -*- coding: utf-8 -*-

"""

"""

__author__ = 'Yngve Mardal Moe'
__email__ = 'yngve.m.moe@gmail.com'

import sys
import numpy as np

import numpy as np
import skimage.data as dt
from perona_malik import *
from matplotlib.pyplot import *
raw = dt.camera().astype('double')/255
raw2 = raw + np.random.random([512, 512])*0.2

imshow(fast_perona_malik(raw2, it=20),
       cmap='gray')


show()
