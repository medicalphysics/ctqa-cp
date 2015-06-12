"""
Created on Wed Dec 04 08:46:48 2013

@author: erlean
"""

import numpy as np
from PyQt4 import QtGui
from _hough_transform import _hough_circle as hough_circle
from _findpeaks import peak_local_max
from _canny_edge import canny
import itertools


colorTable = [QtGui.QColor(i, i, i).rgb() for i in range(256)]
cp404_rodRadii = 1.5  # radii in mm
cp404_minRodSpacing = 40.  # minimum rod spacing
cp404_rodSpacing = 50.  # rod spacing


def pixelArray(dc):
    px = dc.pixel_array.astype(np.int16)
    if 'RescaleSlope' in dc:
        px *= np.int16(dc.RescaleSlope)
    if 'RescaleIntercept' in dc:
        px += np.int16(dc.RescaleIntercept)
    return px


def findCP404Center(array, dxdy, area=False, spacing=False, pos=False):
    rx, ry = findCP404Rods(array, dxdy)
    if len(rx) == 0:
        r = (-1, -1)
        if area:
            r += (-1,)
        if spacing:
            r += (-1,)
        if pos:
            r += (-1, -1)
        return r
    ind = np.hstack((np.arange(len(rx)), 0))
    # Finding center of image and rod spacing
    rod_spacing = 0.
    Cx = 0.
    Cy = 0.
    A = 0.
    for i in range(len(rx)):
        m = rx[ind[i]] * ry[ind[i+1]] - rx[ind[i+1]] * ry[ind[i]]
        A += m
        Cx += (rx[ind[i]] + rx[ind[i+1]]) * m
        Cy += (ry[ind[i]] + ry[ind[i+1]]) * m
        rod_spacing += ((rx[ind[i]] - rx[ind[i + 1]])**2 +
                        (ry[ind[i]] - ry[ind[i + 1]])**2)**.5
    A /= 2.
    Cx /= 6 * A
    Cy /= 6 * A
    rod_spacing /= 4.
    r = (Cx, Cy)
    if area:
        r += (A,)
    if spacing:
        r += (rod_spacing,)
    if pos:
        r += (rx, ry)
    return r


def peak_sort(p, px):
    for ind in itertools.combinations(xrange(p.shape[-1]), 4):
        ind = np.array(ind)
        ang = np.argsort(np.nan_to_num(np.arctan2(p[0, ind] - p[0, ind].mean(),
                         p[1, ind] - p[1, ind].mean())))
        rods = p[:, ind[ang]]
        # test for rods forming a square
        a = rods[:, 0]
        ab = rods[:, 1] - a
        ac = rods[:, 3] - a
        metric = np.sqrt(np.sum((a + ab + ac - rods[:, 2])**2))
        if metric * px > 5.:
            continue
        # test distance
        ind = [0, 1, 2, 3, 0]
        for i in range(4):
            d = np.sum((rods[:, ind[i]] - rods[:, ind[i+1]])**2)**.5
            if np.abs(d - cp404_rodSpacing / px) > 5.:
                continue
        return rods[0, :].ravel(), rods[1, :].ravel()
    return [], []


def findCP404Rods(array, pixel_size, sigma=1.3, ):
    """Attempts to find rods in a image
    Input: array (2D numpy array) with valid hounsfiled units
           pixel_size (tuple of pixel size in x and y dim)
    Output: rod coordinated numpy array with shape (2, 4)
        returns -1 for rod indexes if they are not found
    """
    if sigma:
        if sigma < .3:
            return [], []
    else:
        return [], []

    px = pixel_size

    rod_pixel_spacing = int(cp404_minRodSpacing / px)
    array_th = array #* ((array > 300) + (array < -300))
    edges = canny(array_th, sigma=sigma, low_threshold=300,
                  high_threshold=500)

    hough_radii = np.array([np.round(cp404_rodRadii / px), ]).astype(np.int)
    hough_res = np.max(hough_circle(edges, hough_radii), axis=0)

    peaks = peak_local_max(hough_res, min_distance=rod_pixel_spacing,
                           threshold_abs=.50, num_peaks=30)

    if len(peaks) == 0:
        return [], []

    if peaks.shape[0] < 4:
        return [], []

    if peaks.shape[0] > 8:
        peaks = peaks[:8,:]

    x, y = list(peaks[:, 0]), list(peaks[:, 1])

    # sorting out neighboring peaks
    x_s, y_s = [x.pop(0)], [y.pop(0)]
    while len(x) > 0:
        x_c, y_c = x.pop(0), y.pop(0)
        dist = [((x_c-xi)**2 + (y_c-yi)**2)**.5 > rod_pixel_spacing / 2.
                for xi, yi in zip(x_s, y_s)]
        if all(dist):
            x_s.append(x_c)
            y_s.append(y_c)

    if len(x_s) < 4:
        return [], []

    # returning rods sorted by angle from center
    x_a = np.array(x_s)
    y_a = np.array(y_s)
    return peak_sort(np.vstack((x_a, y_a)), px)


def findCP404Rods_oldVersion(array, pixel_size, sigma=1.3):
    """Attempts to find rods in a image
    Input: array (2D numpy array) with valid hounsfiled units
           pixel_size (tuple of pixel size in x and y dim)
    Output: rod coordinated numpy array with shape (2, 4)
        returns -1 for rod indexes if they are not found
    """
    if sigma:
        if sigma < .3:
            return [], []
    else:
        return [], []

    px = pixel_size

    rod_pixel_spacing = int(cp404_minRodSpacing / px)
    array_th = array * ((array > 300) + (array < -300))
    edges = canny(array_th, sigma=sigma, low_threshold=300,
                  high_threshold=500)

    hough_radii = np.array([np.round(cp404_rodRadii / px), ]).astype(np.int)
    hough_res = np.max(hough_circle(edges, hough_radii), axis=0)

    peaks = peak_local_max(hough_res, min_distance=rod_pixel_spacing,
                           threshold_rel=.45, num_peaks=30)

    if len(peaks) == 0:
        return [], []

    if peaks.shape[0] < 4:
        return [], []

    x, y = list(peaks[:, 0]), list(peaks[:, 1])

    # sorting out neighboring peaks
    x_s, y_s = [x.pop(0)], [y.pop(0)]
    while len(x_s) < 4 and len(x) > 0:
        x_c, y_c = x.pop(0), y.pop(0)
        dist = [((x_c-xi)**2 + (y_c-yi)**2)**.5 > rod_pixel_spacing
                for xi, yi in zip(x_s, y_s)]
        if all(dist):
            x_s.append(x_c)
            y_s.append(y_c)

    if len(x_s) != 4:
        return [], []
    # returning rods sorted by angle from center
    x_a = np.array(x_s)
    y_a = np.array(y_s)
    ang = np.argsort(np.nan_to_num(np.arctan2(x_a - x_a.mean(),
                                              y_a - y_a.mean())))
    rods = np.vstack((x_a, y_a))[:, ang]

    # test for rods forming a square
    a = rods[:, 0]
    ab = rods[:, 1] - a
    ac = rods[:, 3] - a
    metric = np.sqrt(np.sum((a + ab + ac - rods[:, 2])**2))
    if metric * px > 5.:
        return [], []
    # test distance
    ind = [0, 1, 2, 3, 0]
    for i in range(4):
        d = np.sum((rods[:, ind[i]] - rods[:, ind[i+1]])**2)**.5
        if np.abs(d - cp404_rodSpacing / px) > 5.:
            return [], []
    return rods[0, :].ravel(), rods[1, :].ravel()


def arrayToQImage(array, WC=0, WW=500):
    """Convert the 2D numpy array `gray` into a 8-bit QImage with a gray
    colormap.  The first dimension represents the vertical image axis.
    ATTENTION: This QImage carries an attribute `ndimage` with a
    reference to the underlying numpy array that holds the data. On
    Windows, the conversion into a QPixmap does not copy the data, so
    that you have to take care that the QImage does not get garbage
    collected (otherwise PyQt will throw away the wrapper, effectively
    freeing the underlying memory - boom!)."""
    array = (np.clip(array, WC - 0.5 - (WW-1) / 2, WC - 0.5 + (WW - 1) / 2) -
             (WC - 0.5 - (WW - 1) / 2)) * 255 / ((WC - 0.5 + (WW - 1) / 2) -
                                                 (WC - 0.5 - (WW - 1) / 2))
    array = np.require(array, np.uint8, 'C')
    h, w = array.shape
    result = QtGui.QImage(array.data, w, h, QtGui.QImage.Format_Indexed8)
    result.ndarray = array
    result.setColorTable(colorTable)
    return result


def rebin(a, newshape):
    '''Rebin an array to a new shape.
    '''
    assert len(a.shape) == len(newshape)

    slices = [slice(0, old, float(old) / new) for old, new in zip(a.shape,
                                                                  newshape)]
    coordinates = np.mgrid[slices]
    indices = coordinates.astype('i')
    return a[tuple(indices)]


def massCenter(a):
    arr = np.array(a, dtype=np.float)
    totalMass = arr.sum()
    sh = arr.shape
    if totalMass == 0:
        return (0,) * len(sh)
    if len(arr.shape) == 3:
        xcm = np.sum(np.sum(np.sum(arr, axis=2),
                            axis=1) * (np.indices([sh[0]])[0, :]))
        ycm = np.sum(np.sum(np.sum(arr, axis=2),
                            axis=0) * (np.indices([sh[1]])[0, :]))
        zcm = np.sum(np.sum(np.sum(arr, axis=1),
                            axis=0) * (np.indices([sh[2]])[0, :]))
        return xcm / totalMass, ycm / totalMass, zcm / totalMass
    elif len(arr.shape) == 2:
        xcm = np.sum(np.sum(arr, axis=1) * (np.indices([sh[0]])[0, :]))
        ycm = np.sum(np.sum(arr, axis=0) * (np.indices([sh[1]])[0, :]))
        return xcm / totalMass, ycm / totalMass
    elif len(arr.shape) == 1:
        xcm = np.sum(arr * np.indices([sh[0]]))
        return xcm / totalMass
    else:
        return None


def radialSampling(arr, center=None):
    x, y = np.indices(arr.shape)
    if not center:
        center = np.array([(x.max())/2.0, (y.max())/2.0])
    r = np.hypot(x - center[0], y - center[1])
    sort_index = np.argsort(r.flat)
    return r.flat[sort_index], arr.flat[sort_index]


def fft2DRadProf(ft, d=1.):
    coord = np.fft.fftshift(np.fft.fftfreq(ft.shape[0], d))
    image = np.fft.fftshift(ft)

    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    center = np.squeeze(np.array([x[0, coord == 0.0], y[coord == 0.0, 0]]))
    r = np.hypot(x - center[0], y - center[1])
    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = np.rint(r_sorted).astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin

    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr
    cl = np.rint(np.sum(coord >= 0))
    return coord[coord >= 0], radial_prof[:cl]


def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is
             None, which then uses the center of the image (including
             fracitonal pixels).
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)
    if not center:
        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])
    r = np.hypot(x - center[0], y - center[1])
    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]
    radial_prof = tbin / nr
    return radial_prof


def circleMask(array_shape, radius, center=None):
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


def circleIndices(array_shape, radius, center):
    sx, sy = array_shape
    a = np.zeros((sx + radius * 2, sy + radius * 2), np.int)
    cx, cy = center  # The center of circle
    cx += radius
    cy += radius
    x, y = np.ogrid[-radius: radius, -radius: radius]
    index = x**2 + y**2 <= radius**2
    a[cx-radius:cx+radius+1, cy-radius:cy+radius+1][index] = 1
    return np.nonzero(a[radius:-radius, radius:-radius])
