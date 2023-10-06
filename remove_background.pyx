# -*- coding: utf-8 -*-

#python setup.py build_ext --inplace
#import cv2 as cv

import numpy as np
cimport numpy as cnp
import cython

cnp.import_array()
#DTYPE = np.ubyte
ctypedef cnp.npy_ubyte DTYPE_t

print("Background remover has been Imported")

@cython.boundscheck(False)
cpdef cnp.ndarray cRemove_background(cnp.ndarray[DTYPE_t, ndim=3] frame, cnp.ndarray[DTYPE_t, ndim=3] background_image, float percent_match = 0.03 ):
    """ Remove the background image from the current frame
        loops through all pixels, and if their if HSV are similar
        by x%, theyre replaced with all black pixel.
        This is a cythonised method.
        Image in must be greyscaled
    """
    cdef int rows, cols, row, col, min, max
    cdef cnp.ndarray[DTYPE_t, ndim=1] pix_vals
    rows = frame.shape[0]
    cols = frame.shape[1]

    #cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

    # Loop through all pixels
    for row in range(rows):
        for col in range(cols):
            # Get range that pixel should be in to be considered background pix
            min = int(background_image[row, col] * (1 - percent_match))
            max =  int(background_image[row, col]  * (1 + percent_match))
            # Set image to black, if detected as background
            if min < frame[row, col] < max:
                frame[row, col] = 0

    return frame