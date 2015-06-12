
"""
Functions taken from  Luke Campagnola
Distributed under MIT/X11 license.
"""
from PyQt4 import QtGui, QtCore
import numpy as np
import ctypes
import os

Colors = {
    'b': (0, 0, 255, 255),
    'g': (0, 255, 0, 255),
    'r': (255, 0, 0, 255),
    'c': (0, 255, 255, 255),
    'm': (255, 0, 255, 255),
    'y': (255, 255, 0, 255),
    'k': (0, 0, 0, 255),
    'w': (255, 255, 255, 255),
}


def qUrlToStr(qu):
    return qStringToStr(qu.toLocalFile())


def qStringToStr(qs):
#    return unicode(qs).encode(coding).decode(coding)
#    return str(qs.toUtf8()).encode('utf-8')
    return unicode(qs.toUtf8(), encoding="UTF-8")
#    return str(qs.toAscii())
#    return unicode(qs).encode('utf-8')#.decode('utf-8')


def mkColor(*args):
    """
    Convenience function for constructing QColor from a variety of argument
    types. Accepted arguments are:

    ================ ================================================
     'c'             one of: r, g, b, c, m, y, k, w
     R, G, B, [A]    integers 0-255
     (R, G, B, [A])  tuple of integers 0-255
     float           greyscale, 0.0-1.0
     int             see :func:`intColor() <pyqtgraph.intColor>`
     (int, hues)     see :func:`intColor() <pyqtgraph.intColor>`
     "RGB"           hexadecimal strings; may begin with '#'
     "RGBA"
     "RRGGBB"
     "RRGGBBAA"
     QColor          QColor instance; makes a copy.
    ================ ================================================
    """
    err = 'Not sure how to make a color from "%s"' % str(args)
    if len(args) == 1:
        if isinstance(args[0], QtGui.QColor):
            return QtGui.QColor(args[0])
        elif isinstance(args[0], float):
            r = g = b = int(args[0] * 255)
            a = 255
        elif isinstance(args[0], basestring):
            c = args[0]
            if c[0] == '#':
                c = c[1:]
            if len(c) == 1:
                (r, g, b, a) = Colors[c]
            if len(c) == 3:
                r = int(c[0]*2, 16)
                g = int(c[1]*2, 16)
                b = int(c[2]*2, 16)
                a = 255
            elif len(c) == 4:
                r = int(c[0]*2, 16)
                g = int(c[1]*2, 16)
                b = int(c[2]*2, 16)
                a = int(c[3]*2, 16)
            elif len(c) == 6:
                r = int(c[0:2], 16)
                g = int(c[2:4], 16)
                b = int(c[4:6], 16)
                a = 255
            elif len(c) == 8:
                r = int(c[0:2], 16)
                g = int(c[2:4], 16)
                b = int(c[4:6], 16)
                a = int(c[6:8], 16)
        elif hasattr(args[0], '__len__'):
            if len(args[0]) == 3:
                (r, g, b) = args[0]
                a = 255
            elif len(args[0]) == 4:
                (r, g, b, a) = args[0]
            elif len(args[0]) == 2:
                return intColor(*args[0])
            else:
                raise Exception(err)
        elif type(args[0]) == int:
            return intColor(args[0])
        else:
            raise Exception(err)
    elif len(args) == 3:
        (r, g, b) = args
        a = 255
    elif len(args) == 4:
        (r, g, b, a) = args
    else:
        raise Exception(err)

    args = [r, g, b, a]
    args = [0 if np.isnan(a) or np.isinf(a) else a for a in args]
    args = list(map(int, args))
    return QtGui.QColor(*args)


def mkBrush(*args, **kwds):
    """
    | Convenience function for constructing Brush.
    | This function always constructs a solid brush and accepts the same
      arguments as :func:`mkColor() <pyqtgraph.mkColor>`
    | Calling mkBrush(None) returns an invisible brush.
    """
    if 'color' in kwds:
        color = kwds['color']
    elif len(args) == 1:
        arg = args[0]
        if arg is None:
            return QtGui.QBrush(QtCore.Qt.NoBrush)
        elif isinstance(arg, QtGui.QBrush):
            return QtGui.QBrush(arg)
        else:
            color = arg
    elif len(args) > 1:
        color = args
    return QtGui.QBrush(mkColor(color))


def mkPen(*args, **kargs):
    """
    Convenience function for constructing QPen.

    Examples::

        mkPen(color)
        mkPen(color, width=2)
        mkPen(cosmetic=False, width=4.5, color='r')
        mkPen({'color': "FF0", width: 2})
        mkPen(None)   # (no pen)

    In these examples, *color* may be replaced with any arguments accepted by :
    func:`mkColor() <pyqtgraph.mkColor>`    """

    color = kargs.get('color', None)
    width = kargs.get('width', 1)
    style = kargs.get('style', None)
    cosmetic = kargs.get('cosmetic', True)
    hsv = kargs.get('hsv', None)

    if len(args) == 1:
        arg = args[0]
        if isinstance(arg, dict):
            return mkPen(**arg)
        if isinstance(arg, QtGui.QPen):
            return QtGui.QPen(arg)  # return a copy of this pen
        elif arg is None:
            style = QtCore.Qt.NoPen
        else:
            color = arg
    if len(args) > 1:
        color = args

    if color is None:
        color = mkColor(200, 200, 200)
    if hsv is not None:
        color = hsvColor(*hsv)
    else:
        color = mkColor(color)

    pen = QtGui.QPen(QtGui.QBrush(color), width)
    pen.setCosmetic(cosmetic)
    if style is not None:
        pen.setStyle(style)
    return pen


def hsvColor(hue, sat=1.0, val=1.0, alpha=1.0):
    """Generate a QColor from HSVa values. (all arguments are float 0.0-1.0)"""
    c = QtGui.QColor()
    c.setHsvF(hue, sat, val, alpha)
    return c


def colorTuple(c):
    """Return a tuple (R,G,B,A) from a QColor"""
    return (c.red(), c.green(), c.blue(), c.alpha())


def colorStr(c):
    """Generate a hex string code from a QColor"""
    return ('%02x'*4) % colorTuple(c)


def intColor(index, hues=9, values=1, maxValue=255, minValue=150, maxHue=360,
             minHue=0, sat=255, alpha=255, **kargs):
    """
    Creates a QColor from a single index. Useful for stepping through a
    predefined list of colors.

    The argument *index* determines which color from the set will be returned.
    All other arguments determine what the set of predefined colors will be

    Colors are chosen by cycling across hues while varying the value
    (brightness).
    By default, this selects from a list of 9 hues."""
    hues = int(hues)
    values = int(values)
    ind = int(index) % (hues * values)
    indh = ind % hues
    indv = ind / hues
    if values > 1:
        v = minValue + indv * ((maxValue-minValue) / (values-1))
    else:
        v = maxValue
    h = minHue + (indh * (maxHue-minHue)) / hues

    c = QtGui.QColor()
    c.setHsv(h, sat, v)
    c.setAlpha(alpha)
    return c


def makeArrowPath(headLen=20, tipAngle=20, tailLen=20, tailWidth=3,
                  baseAngle=0):
    """
    Construct a path outlining an arrow with the given dimensions.
    The arrow points in the -x direction with tip positioned at 0,0.
    If *tipAngle* is supplied (in degrees), it overrides *headWidth*.
    If *tailLen* is None, no tail will be drawn.
    """
    headWidth = headLen * np.tan(tipAngle * 0.5 * np.pi/180.)
    path = QtGui.QPainterPath()
    path.moveTo(0, 0)
    path.lineTo(headLen, -headWidth)
    if tailLen is None:
        innerY = headLen - headWidth * np.tan(baseAngle*np.pi/180.)
        path.lineTo(innerY, 0)
    else:
        tailWidth *= 0.5
        innerY = headLen - (headWidth-tailWidth) * np.tan(baseAngle*np.pi/180.)
        path.lineTo(innerY, -tailWidth)
        path.lineTo(headLen + tailLen, -tailWidth)
        path.lineTo(headLen + tailLen, tailWidth)
        path.lineTo(innerY, tailWidth)
    path.lineTo(headLen, headWidth)
    path.lineTo(0, 0)
    return path


def rescaleData(data, scale, offset, dtype=None):
    """Return data rescaled and optionally cast to a new dtype::
       data => (data-offset) * scale
    """
    if dtype is None:
        dtype = data.dtype

    d2 = data-offset
    d2 *= scale
    data = d2.astype(dtype)
    return data


def applyLookupTable(data, lut):
    """
    Uses values in *data* as indexes to select values from *lut*.
    The returned data has shape data.shape + lut.shape[1:]
    """
    if data.dtype.kind not in ('i', 'u'):
        data = data.astype(int)
    return np.take(lut, data, axis=0, mode='clip')


def makeARGB(data, lut=None, levels=None, scale=None):
    """
    Convert an array of values into an ARGB array suitable for building
    QImages, OpenGL textures, etc.

    Returns the ARGB array (values 0-255) and a boolean indicating whether
    there is alpha channel data.
    This is a two stage process:

        1) Rescale the data based on the values in the *levels* argument
           (min, max).
        2) Determine the final output by passing the rescaled values through a
           lookup table.

    Both stages are optional.

    ============ ==================================================================================
    Arguments:
    data         numpy array of int/float types. If
    levels       List [min, max]; optionally rescale data before converting through the
                 lookup table. The data is rescaled such that min->0 and max->*scale*::

                    rescaled = (clip(data, min, max) - min) * (*scale* / (max - min))

                 It is also possible to use a 2D (N,2) array of values for levels. In this case,
                 it is assumed that each pair of min,max values in the levels array should be
                 applied to a different subset of the input data (for example, the input data may
                 already have RGB values and the levels are used to independently scale each
                 channel). The use of this feature requires that levels.shape[0] == data.shape[-1].
    scale        The maximum value to which data will be rescaled before being passed through the
                 lookup table (or returned if there is no lookup table). By default this will
                 be set to the length of the lookup table, or 256 is no lookup table is provided.
                 For OpenGL color specifications (as in GLColor4f) use scale=1.0
    lut          Optional lookup table (array with dtype=ubyte).
                 Values in data will be converted to color by indexing directly from lut.
                 The output data shape will be input.shape + lut.shape[1:].

                 Note: the output of makeARGB will have the same dtype as the lookup table, so
                 for conversion to QImage, the dtype must be ubyte.

                 Lookup tables can be built using GradientWidget.
    useRGBA      If True, the data is returned in RGBA order (useful for building OpenGL textures).
                 The default is False, which returns in ARGB order for use with QImage
                 (Note that 'ARGB' is a term used by the Qt documentation; the _actual_ order
                 is BGRA).
    ============ ==================================================================================
    """

    if lut is not None and not isinstance(lut, np.ndarray):
        lut = np.array(lut)
    if levels is not None and not isinstance(levels, np.ndarray):
        levels = np.array(levels)

    if levels is not None:
        if levels.ndim == 1:
            if len(levels) != 2:
                raise Exception('levels argument must have length 2')
        elif levels.ndim == 2:
            if lut is not None and lut.ndim > 1:
                raise Exception('Cannot make ARGB data when bot levels and lut have ndim > 2')
            if levels.shape != (data.shape[-1], 2):
                raise Exception('levels must have shape (data.shape[-1], 2)')
        else:
#            print levels
            raise Exception("levels argument must be 1D or 2D.")

    if scale is None:
        if lut is not None:
            scale = lut.shape[0]
        else:
            scale = 255.

    ## Apply levels if given
    if levels is not None:

        if isinstance(levels, np.ndarray) and levels.ndim == 2:
            ## we are going to rescale each channel independently
            if levels.shape[0] != data.shape[-1]:
                raise Exception("When rescaling multi-channel data, there must be the same number of levels as channels (data.shape[-1] == levels.shape[0])")
            newData = np.empty(data.shape, dtype=int)
            for i in range(data.shape[-1]):
                minVal, maxVal = levels[i]
                if minVal == maxVal:
                    maxVal += 1e-16
                newData[..., i] = rescaleData(data[..., i],
                                              scale / (maxVal - minVal),
                                              minVal, dtype=int)
            data = newData
        else:
            minVal, maxVal = levels
            if minVal == maxVal:
                maxVal += 1e-16
            data = rescaleData(data, scale / (maxVal - minVal),
                               minVal, dtype=int)

    ## apply LUT if given
    if lut is not None:
        data = applyLookupTable(data, lut)
    else:
        if data.dtype is not np.ubyte:
            data = np.clip(data, 0, 255).astype(np.ubyte)

    ## copy data into ARGB ordered array
    imgData = np.empty(data.shape[:2]+(4,), dtype=np.ubyte)
    if data.ndim == 2:
        data = data[..., np.newaxis]

    order = [2, 1, 0, 3]  # for some reason, the colors line up as
                          # BGR in the final image.

    if data.shape[2] == 1:
        for i in range(3):
            imgData[..., order[i]] = data[..., 0]
    else:
        for i in range(0, data.shape[2]):
            imgData[..., order[i]] = data[..., i]

    if data.shape[2] == 4:
        alpha = True
    else:
        alpha = False
        imgData[..., 3] = 255

    return imgData, alpha


def makeQImage(imgData, alpha=None, copy=True, transpose=True):
    """
    Turn an ARGB array into QImage.
    By default, the data is copied; changes to the array will not
    be reflected in the image. The image will be given a 'data' attribute
    pointing to the array which shares its data to prevent python
    freeing that memory while the image is in use.

    =========== ===================================================================
    Arguments:
    imgData     Array of data to convert. Must have shape (width, height, 3 or 4)
                and dtype=ubyte. The order of values in the 3rd axis must be
                (b, g, r, a).
    alpha       If True, the QImage returned will have format ARGB32. If False,
                the format will be RGB32. By default, _alpha_ is True if
                array.shape[2] == 4.
    copy        If True, the data is copied before converting to QImage.
                If False, the new QImage points directly to the data in the array.
                Note that the array must be contiguous for this to work.
    transpose   If True (the default), the array x/y axes are transposed before
                creating the image. Note that Qt expects the axes to be in
                (height, width) order whereas pyqtgraph usually prefers the
                opposite.
    =========== ===================================================================
    """
    ## create QImage from buffer

    ## If we didn't explicitly specify alpha, check the array shape.
    if alpha is None:
        alpha = (imgData.shape[2] == 4)

    copied = False
    if imgData.shape[2] == 3:  # need to make alpha channel
                               # (even if alpha==False; QImage requires 32 bpp)
        if copy is True:
            d2 = np.empty(imgData.shape[:2] + (4,), dtype=imgData.dtype)
            d2[:, :, :3] = imgData
            d2[:, :, 3] = 255
            imgData = d2
            copied = True
        else:
            raise Exception('Array has only 3 channels; cannot make QImage without copying.')

    if alpha:
        imgFormat = QtGui.QImage.Format_ARGB32
    else:
        imgFormat = QtGui.QImage.Format_RGB32

    if transpose:
        imgData = imgData.transpose((1, 0, 2))  ## QImage expects the row/column order to be opposite

    if not imgData.flags['C_CONTIGUOUS']:
        if copy is False:
            extra = ' (try setting transpose=False)' if transpose else ''
            raise Exception('Array is not contiguous; cannot make QImage without copying.'+extra)
        imgData = np.ascontiguousarray(imgData)
        copied = True

    if copy is True and copied is False:
        imgData = imgData.copy()

    addr = ctypes.addressof(ctypes.c_char.from_buffer(imgData, 0))
    img = QtGui.QImage(addr, imgData.shape[1], imgData.shape[0], imgFormat)
    img.data = imgData
    return img


def arrayToQImage(array, WC=0, WW=500):
    argb = makeARGB(array, levels=[WC - WW, WC + WW])
    return makeQImage(argb[0], copy=True, transpose=False)


def imageToArray(img, copy=False, transpose=True):
    """
    Convert a QImage into numpy array. The image must have format RGB32, ARGB32, or ARGB32_Premultiplied.
    By default, the image is not copied; changes made to the array will appear in the QImage as well (beware: if
    the QImage is collected before the array, there may be trouble).
    The array will have shape (width, height, (b,g,r,a)).
    """
    fmt = img.format()
    ptr = img.bits()

    ptr.setsize(img.byteCount())
    arr = np.asarray(ptr)

    if fmt == img.Format_RGB32:
        arr = arr.reshape(img.height(), img.width(), 3)
    elif fmt == img.Format_ARGB32 or fmt == img.Format_ARGB32_Premultiplied:
        arr = arr.reshape(img.height(), img.width(), 4)

    if copy:
        arr = arr.copy()

    if transpose:
        return arr.transpose((1, 0, 2))
    else:
        return arr


def colorToAlpha(data, color):
    """
    Given an RGBA image in *data*, convert *color* to be transparent.
    *data* must be an array (w, h, 3 or 4) of ubyte values and *color* must be
    an array (3) of ubyte values.
    This is particularly useful for use with images that have a black or white background.

    Algorithm is taken from Gimp's color-to-alpha function in plug-ins/common/colortoalpha.c
    Credit:
        /*
        * Color To Alpha plug-in v1.0 by Seth Burgess, sjburges@gimp.org 1999/05/14
        *  with algorithm by clahey
        */

    """
    data = data.astype(float)
    if data.shape[-1] == 3:  # add alpha channel if needed
        d2 = np.empty(data.shape[:2]+(4,), dtype=data.dtype)
        d2[..., :3] = data
        d2[..., 3] = 255
        data = d2

    color = color.astype(float)
    alpha = np.zeros(data.shape[:2]+(3,), dtype=float)
    output = data.copy()

    for i in [0, 1, 2]:
        d = data[..., i]
        c = color[i]
        mask = d > c
        alpha[..., i][mask] = (d[mask] - c) / (255. - c)
        imask = d < c
        alpha[..., i][imask] = (c - d[imask]) / c

    output[..., 3] = alpha.max(axis=2) * 255.

    # avoid zero division while processing alpha channel
    mask = output[..., 3] >= 1.0
    # increase value to compensate for decreased alpha
    correction = 255. / output[..., 3][mask]
    for i in [0, 1, 2]:
        output[..., i][mask] = ((output[..., i][mask]-color[i]) * correction)
        output[..., i][mask] += color[i]
        # combine computed and previous alpha values
        output[..., 3][mask] *= data[..., 3][mask] / 255.

    return np.clip(output, 0, 255).astype(np.ubyte)
