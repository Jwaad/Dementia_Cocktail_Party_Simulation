# -*- coding: utf-8 -*-

#python setup.py build_ext --inplace

import cv2
import numpy as np
cimport numpy as cnp
import cython

cnp.import_array()
DTYPE = np.uint8
ctypedef cnp.uint8_t DTYPE_t

print("Background remover has been Imported")

@cython.boundscheck(False)
cpdef cnp.ndarray cRemove_background(cnp.ndarray[DTYPE_t, ndim=3] frame, cnp.ndarray[DTYPE_t, ndim=3] background_image, float percent_match = 0.05 ):
    """ Remove the background image from the current frame
        loops through all pixels, and if their if HSV are similar
        by x%, theyre replaced with all black pixel.
        This is a cythonised method.
        Image in must be greyscaled
    """
    cdef int rows, cols, row, col, pix_vals, val
    cdef int skip # use int instead of bool
    cdef double min_scalar, max_scalar, min, max
    cdef cnp.ndarray[DTYPE_t, ndim=2] mask

    # Pre calculate the scalar for pix min and max. E.G.: if match = 0.1 -> min = 0.9 max = 1.1
    min_scalar = (1.0 - percent_match)
    max_scalar = (1.0 + percent_match)

    rows = frame.shape[0]
    cols = frame.shape[1]
    pix_vals = frame.shape[2]
    mask = np.zeros((rows,cols),dtype=DTYPE)
    
    # Convert to HSV for colour comparison
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2HSV)

    # Loop through all pixels
    for row in range(rows):
        for col in range(cols):
            skip = False
            for val in range(pix_vals):
                # Get range that pixel should be in to be considered background pix
                min = (background_image[row, col, val] * min_scalar)
                max = (background_image[row, col, val]  * max_scalar)
                val = frame[row, col, val]
                # If any of the 3 HSV values aren't in range, skip pixel
                if not min < val < max:
                    skip = True
                    break
            # Set image to black, if detected as background
            if skip:                
                mask[row, col] = 255
            else:
                mask[row, col] = 0
    
    # Convert back to bgr
    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    background_image = cv2.cvtColor(background_image, cv2.COLOR_HSV2BGR)

    # Finally: erode and dilate our mask (to make it less noisy)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=3)
    mask = cv2.dilate(mask, kernel, iterations=5)

    # Apply the mask to the original image
    frame = cv2.bitwise_and(frame, frame, mask= mask)

    return frame