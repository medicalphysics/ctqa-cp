# -*- coding: utf-8 -*-

import numpy as np
from scipy.ndimage import measurements, binary_erosion
from ctqa.utils import image_world_transform
import itertools 
import logging
logger = logging.getLogger('CTQA')
from matplotlib import pylab as plt



def circle_mask(array_shape, radius, center=None):
    a = np.zeros(array_shape, np.int)
    if center is None:
        cx = array_shape[0] / 2
        cy = array_shape[1] / 2
    else:
        cx, cy = center
    y, x = np.ogrid[-radius: radius, -radius: radius]
    index = x**2 + y**2 <= radius**2
    a[cx-radius:cx+radius+1, cy-radius:cy+radius+1][index] = 1
    return a
    
    
def is_corners(points, dist=50):
    center = points.sum(axis=0) / 4.
    
    p_r = points - np.repeat(center.reshape(-1, 2), 4, axis=0)
    
    #test for eqvidistance
    diff = .05 # allows for 5 percent diff   
    center_dist = np.sum(p_r**2, axis=1)**.5
    center_dist_req = dist / np.sqrt(2)
    
#    import pdb;pdb.set_trace()
    if not np.all((center_dist > center_dist_req*(1-diff))*(center_dist < center_dist_req*(1+diff))):
        return False, None
    

    
    #sorting points
    sort_ind = np.argsort(np.arctan2(p_r[:, 0], p_r[:, 1]))
    v = points[sort_ind, :]
    #comparing calculated vs actula position of point 2
    
    p2 = v[1, :] + v[3, :] - v[0, :]
#    import pdb;pdb.set_trace()
    if np.sum((p2-v[2,:])**2)**.5 > dist*.01:
        return False, None
    return True, v
    
    
def relative_CP404_position(array, corners, shape, pixel_spacing):

    corners_int = np.rint(corners).astype(np.int)
    
    dist20 = int(round(20. /pixel_spacing[1]))
    dist5 = int(round(5. /pixel_spacing[1]))
    # test if regions inside image
    for dist in [-dist20, dist20]:
        for i in range(4):
            for j in range(2):
                if not (0 <= corners[i, j] + dist < shape[j+1]):
                    return False, None
        
    
    region_x1 = array[corners_int[0, 0]-dist20 : corners_int[0, 0] - dist5, corners_int[0, 1] : corners_int[1, 1]].max(axis=0)
    region_y1 = array[corners_int[1, 0]: corners_int[2, 0], corners_int[1, 1] + dist5 : corners_int[1, 1] + dist20].max(axis=1)
    region_x2 = array[corners_int[2, 0] + dist5 : corners_int[2, 0] + dist20, corners_int[3, 1] : corners_int[2, 1]].max(axis=0)
    region_y2 = array[corners_int[0, 0]: corners_int[3, 0], corners_int[3, 1] - dist20 : corners_int[3, 1]-dist5].max(axis=1)
    if not all([reg.max() > 700 for reg in [region_x1, region_x2, region_y1, region_y2]]):
        return False, None
    pos = np.empty(4)
#    import pdb;pdb.set_trace()
    pos[0] = np.sum(np.arange(region_x1.shape[0]) * region_x1) / region_x1.sum() - region_x1.shape[0] / 2.
    pos[1] = np.sum(np.arange(region_x2.shape[0]) * region_x2[::-1]) / region_x2.sum() - region_x2.shape[0] / 2.
    pos[2] = np.sum(np.arange(region_y1.shape[0]) * region_y1) / region_y1.sum() - region_y1.shape[0] / 2.
    pos[3] = np.sum(np.arange(region_y2.shape[0]) * region_y2[::-1]) / region_y2.sum() - region_y2.shape[0] / 2.
    factor = pixel_spacing[1] / np.tan(np.deg2rad(23.))
    if (pos.std() * factor) > 2.:
        return False, None
    mean_pos = pos.mean() * factor
    
    if -8 <= mean_pos <= 8:
        return True, mean_pos
    return False, None
    
    
def is_image_in_CP404(array, shape, pixel_spacing):
    array = np.squeeze(array)
    
    ## finding regions of air or teflon that has equal area to the air/teflon rods 
    b = (array > 700) + (array < -700)
    l, nlabels = measurements.label(b)
    
    ones = np.ones_like(l)
    areas = measurements.sum(ones, labels=l, index=np.arange(nlabels))
    target_area = np.pi * ( 0.5 * np.array([2.0, 3.25]))**2 / pixel_spacing[1:].prod()
    labels_of_interest = np.arange(nlabels, dtype=np.int)[(areas >= target_area[0]) * (areas <= target_area[1])]

    #testing if we found more than 4 rods
    if len(labels_of_interest) < 4:
        return False, None, None

    dist = 50. / pixel_spacing[1]    
    
    ## testing if any combinations of the rods form a square
    for ind, points in enumerate(itertools.combinations(labels_of_interest, 4)):
        coords = measurements.center_of_mass(ones, l, points)
        
        found, corners = is_corners(np.array(coords), dist=dist)
        if found:
            break
        elif ind > 8: # max number of combinations to test
            return False, None, None
    else:
        return False, None, None

    ##obtains localization of the image relative to the phantom    
    success, pos = relative_CP404_position(array, corners, shape, pixel_spacing)
    if success:
        return True, corners, pos
    return False, None, None
    
def localize(database):
    """example function to localize all images in database
    """
    #selecting FORs
    fors = set(database.get_metadata('for_uid', condition='is_localized==False'))   
#    fors = set(database.get_metadata('for_uid', condition=''))   
    for for_uid in fors:
        logger.debug('Localizing frame of reference {}'.format(for_uid))
        series = database.get_series(condition='for_uid == b"{}"'.format(for_uid))    
        position = []
        measured_pos = []
        world_corners = []
        for series_uid in series:
            logger.debug('Localizer is processing series {}'.format(series_uid))
            metadata = database.get_series_metadata(series_uid)
            study_uid = metadata['study_uid']
            pixel_spacing = metadata['spacing']
            image_position = metadata['image_position']
            image_orientation = metadata['image_orientation']
            shape = metadata['shape']
            image_positions = database.get_series_array(study_uid, series_uid, 'image_positions')
            
            for i in range(shape[0]):
                array = database.get_series_array(study_uid, series_uid, 'ctarray', start=i)
                success, corner, pos = is_image_in_CP404(array, shape, pixel_spacing)
                
                if success:
                    position.append(image_positions[i, 2])
                    measured_pos.append(pos)
#                    world_corners.append(image_world_transform(image_orientation, image_positions[i, :], pixel_spacing, np.hstack((corner, i + np.zeros((4, 1)))) ))
                    world_corners.append(image_world_transform(image_orientation, image_position, pixel_spacing, np.hstack((corner, i + np.zeros((4, 1)))) ))
#                    import pdb; pdb.set_trace()       
            if len(measured_pos) > 30:
                if (max(position) - min(position)) > 10:
                    break
        if len(measured_pos) > 2:
#            plt.plot(position, measured_pos,'o')
#            plt.show()
#            import pdb; pdb.set_trace()
            offset = np.linalg.lstsq(np.vstack([np.array(position), np.ones(len(position))]).T, np.array(measured_pos))[0][1]
            corners = np.array(world_corners).mean(axis=0)
#            import pdb; pdb.set_trace()       
            corners[:, 2] = offset
            fields = {'is_localized': True, 
                      'phantom_offset': offset,
                      'cp404_rods_position': corners.ravel(),
                      }
            for series_uid in series:
                database.update_series_metadata(series_uid, fields)
                logger.debug('Updating position offset for series {}'.format(series_uid))

    