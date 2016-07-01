# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 10:05:18 2016

@author: erlean
"""
import numpy as np

def image_world_transform(image_orientation, image_position, pixel_spacing, indices, inverse=False):    
    x = np.array(image_orientation[3:])
    y = np.array(image_orientation[:3])
    z = np.cross(y, x)
    A = np.zeros((4, 4))
    A[:3, 0] = x * pixel_spacing[1]
    A[:3, 1] = y * pixel_spacing[2]
    A[:3, 2] = z * pixel_spacing[0]
    A[:3, 3] = image_position
    A[3, 3] = 1
    if inverse:
        A = np.linalg.inv(A)
#    import pdb;pdb.set_trace()
    return np.dot(A, np.vstack((indices.T, np.ones(indices.shape[0])))  )[:3, :].T
    

def circle_mask(array_shape, radius, center=None):
    a = np.zeros(array_shape, np.int)
    if not center:
        cx = array_shape[0] / 2
        cy = array_shape[1] / 2
    else:
        cx, cy = center
    y, x = np.ogrid[-radius: radius, -radius: radius]
    index = x**2 + y**2 <= radius**2
    a[cx-radius:cx+radius+1, cy-radius:cy+radius+1][index] = 1
    return a


def circle_indices(array_shape, radius, center):
    sx, sy = array_shape
    cx, cy = center  # The center of circle
    y, x = np.ogrid[-cx:sx-cx,-cy:sy-cy]
    return x*x + y*y <= radius*radius
    
