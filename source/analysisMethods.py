"""
Created on Thu Jan 02 13:38:21 2014

@author: erlean
"""

import numpy as np
import os
import sys
import traceback
from functools import wraps
import imageTools
from _findpeaks import peak_local_max
import math
from lcsFitter import fitLowContrastScore
from attinuationCoef import AttCoef as wideAttCoef

if 140 in wideAttCoef['kev']:
    lim_ind = wideAttCoef['kev'].index(150)
    AttCoef = {}
    for key, value in wideAttCoef.items():
        AttCoef[key] = value[0:lim_ind]
else:
    AttCoef = wideAttCoef


def logger(func):
    @wraps(func)
    def inner(*args, **kwargs):
        try:
            res = func(*args, **kwargs)
        except Exception, e:
            wrn = os.linesep
            wrn += "OUCH, an error occurred. Please forward this message to"
            wrn += os.linesep
            wrn += "the developers (andersen.erlend@gmail.com):"
            wrn += os.linesep
            wrn += "'{0}' in function {1}.{2}".format(e.message,
                                                      func.__module__,
                                                      func.__name__)

            wrn += os.linesep
            wrn += ''.join(traceback.format_exception(*sys.exc_info()))
            wrn += os.linesep
            name = '{0}: error'.format(func.__name__)
            res = Analysis(name, args[-1])
            res.success = False
            res.message = wrn
        finally:
            return res
    return inner


class Analysis(object):
    def __init__(self, name, ID=None):
        self.name = name
        self.ID = ID
        self.success = True
        self.message = ""
        self.warning = ""
        self.imageUids = []
        self.graphicsItems = {}  # {title:('circle', posx,posy, r, r)}
                                 # {title:('line', posx,posy, posx,posy)}
                                 # {title:('rect', posx,posy, w, h)}
        self.graphicsItemsLabelInside = {}  # {title:False} for label outside
        self.images = {}  # {'images':list of numpy arrays, 'pos', list of pos}
        self.dataTable = []  # nested table
        self.plots = {}  # {'title':{'x':x, 'y':y, 'yerr':yerr, 'dots':False,
                         #  'size': 1, 'color':*see pg.mkcolor*,
                         #  'plotText': list of tuples i.e. [(x, y, text),]
                         #  'plotTextAlignment': 'center' or 'upperLeft' or
                         #     'upperRight' or 'lowerLeft' or 'lowerRight'
                         #     'centerLeft' or 'centerRight', }}
        self.plotLabels = {0: 'yLeft', 1: 'yRight', 2: 'xBottom', 3: 'xTop'}
        self.plotLimits = {}  # {0: (0,1), 1: (-100, 100), etc...
        self.plotLegendPosition = None  # 'lowerRight' or 'lowerLeft' or
                                        # 'upperRight' or
                                        # 'upperLeft'


@logger
def pixelSize(arrays, uids, pos, dz, dxdy, orientation, centerInd, ID):
    results = Analysis('Pixel Size', ID)
    array = arrays[0]
    results.imageUids = [uids[0]]
    rx, ry = imageTools.findCP404Rods(array, dxdy)
    if len(rx) == 0:
        results.success = False
        results.message = "Could not locate the air/teflon rods."
        return results
    results.images = {'images': [array], 'pos': [str(p) for p in pos]}
    ind = [0, 1, 2, 3, 0]
    lenghts = []
    label = ['Line ' + str(i) for i in range(1, 5)]
    for i in range(4):
        lenghts.append(np.sqrt((rx[ind[i]] - rx[ind[i + 1]])**2. +
                       (ry[ind[i]] - ry[ind[i + 1]])**2.))

        results.graphicsItems[label[i]] = ('line', ry[ind[i]], rx[ind[i]],
                                           ry[ind[i + 1]], rx[ind[i + 1]])
    dataTable = [['', 'Line 1', 'Line 2', 'Line 3', 'Line 4', 'Mean',
                  'Nominal']]
    dataTable.append(['Pixel size [mm]'])
    dataTable.append(['Deviation [%]'])
    nominal = dxdy
    nlenght = 50.
    mean = nlenght / sum(lenghts) * 4.
    for i in range(4):
        val = np.round(nlenght / lenghts[i], 3)
        dev = np.round((nlenght / lenghts[i] - dxdy) / dxdy * 100., 3)
        dataTable[1].append(format(val, ' .3f'))
        dataTable[2].append(format(dev, ' .3f'))
    dataTable[1].append(format(np.round(mean, 3), ' .3f'))
    dataTable[1].append(format(np.round(nominal, 3), ' .3f'))
    dataTable[2].append(format((mean - dxdy) / dxdy * 100., ' .2f'))
    results.dataTable = dataTable
    return results


def HUlinarityEffEnergy(HUmeas):
    def HUcalc(matName, ikev):
        return np.rint(1000 * (AttCoef[matName][ikev] /
                       AttCoef['Water'][ikev] - 1.))
    rsq = []
    for i in range(len(AttCoef['kev'])):
        m = []
        n = []
        for tag, value in HUmeas.items():
            m.append(HUmeas[tag])
            n.append(HUcalc(tag, i))
        A = np.vstack([n, np.ones(len(n))]).T
        residual = np.linalg.lstsq(A, m)[1]
        rsq.append(residual)

    EkevInd = rsq.index(min(rsq))
    HUeff = {}
    for tag, value in HUmeas.items():
        HUeff[tag] = HUcalc(tag, EkevInd)
    return AttCoef['kev'][EkevInd], HUeff


@logger
def HUlinarity(arrays, uids, pos, dz, dxdy, orientation, centerInd, ID):
    results = Analysis('CT linearity', ID)
    results.imageUids = [uids[0]]
    array = arrays[0]
    Cx, Cy, rod_spacing, rx, ry = imageTools.findCP404Center(array, dxdy,
                                                             spacing=True,
                                                             pos=True)
#    print Cx, Cy, rod_spacing, rx, ry
    if Cx == -1:
        results.success = False
        results.message = "Could not locate the air/teflon rods."
        return results
    angles = [0., 60., 90., 120., 180., 240., 270., 300.]
    roi_radius = np.rint(4.0 / dxdy)
    # diameter between rois, even if pixel spacing is off
    roi_spacing = rod_spacing * 117. / 50. / 2.
    # rotation angle
    try:
        roiAngle = np.arctan(float(rx[1] - rx[0]) / float(ry[1] - ry[0]))
    except ZeroDivisionError:
        roiAngle = 0.

    results.images = {'images': arrays, 'pos': pos}

    HUnom = {'Air': -1000, 'PMP': -200, 'LDPE': -100, 'Polystyrene': -35,
             'Water': 0, 'Acrylic': 120, 'Delrin': 340, 'Teflon': 990}
    HUmean = {}
    HUstd = {}
    measurements = []
    measurements_sd = []
    graphicsItems = []
    sh = array.shape
    for theta_deg in angles:
        theta = np.deg2rad(theta_deg)
        cx = Cx + np.rint(roi_spacing * np.sin(theta + roiAngle))
        cy = Cy + np.rint(roi_spacing * np.cos(theta + roiAngle))
        #  returns error result if sample was not found
        if cx < 0 or not cx < sh[0] or cy < 0 or not cy < sh[1]:
            for x, y in zip(rx, ry):
                results.graphicsItems[str((x, y))] = ('circle', y - roi_radius,
                                                      x - roi_radius,
                                                      roi_radius * 2,
                                                      roi_radius * 2)
                msg = "Unable to find phantom samples"
            results.graphicsItems[msg] = ('rect', sh[1] / 2 - 56,
                                          sh[0] / 2 - 56, 128, 128)
            return results

        graphicsItems.append(('circle', cy - roi_radius, cx - roi_radius,
                              roi_radius * 2, roi_radius * 2))
        roi = array[imageTools.circleIndices(sh, roi_radius,
                                             (cx, cy))]
        measurements.append(np.mean(roi))
        measurements_sd.append(np.std(roi))

    indWater = angles.index(270.0)
    if measurements[indWater] < -300:
        del measurements[indWater]
        del HUnom['Water']
        del graphicsItems[indWater]
        del measurements_sd[indWater]
    # guessing material type for each roi by sorting
    values = sorted(HUnom.items(), key=lambda x: x[1])
    sortind = np.argsort(measurements)
    i = 0
    for key, value in values:
        HUmean[key] = measurements[sortind[i]]
        HUstd[key] = measurements_sd[sortind[i]]
        results.graphicsItems[key] = graphicsItems[sortind[i]]
        results.graphicsItemsLabelInside[key] = False
        i += 1
    # brute force least squares value assignment for material guessing
    # (sorting is better)
#    while len(values) > 0:
#        argminmat=np.empty((len(values),len(measurements) ))
#        for i in range(len(values)):
#            for j in range(len(measurements)):
#                argminmat[i,j]=(values[i][1]-measurements[j])**2.0
#        index=np.unravel_index(np.argmin(argminmat),argminmat.shape)
#        key,value=values.pop(index[0])
#        HUmean[key]=measurements.pop(index[1])
#        HUstd[key]=measurements_sd.pop(index[1])
#        HUbox[key]=graphicsBoxes.pop(index[1])

    # linear regression
    try:
        ekev, HUeff = HUlinarityEffEnergy(HUmean)
    except KeyError:
        results.success = False
        return results

    x = []
    y = []
    ysd = []
    curveLabels = []
    for key, value in sorted(HUeff.items(), key=lambda x: x[1]):
        x.append(value)
        y.append(HUmean[key])
        ysd.append(HUstd[key])
        curveLabels.append((value, HUmean[key], key))
    A = np.vstack([x, np.ones(len(x))]).T
    regressionCoef = np.linalg.lstsq(A, y)[0]
    x = np.array(x)
    y = np.array(y)
    Rsqr = (np.corrcoef(y,
                        x * regressionCoef[0] + regressionCoef[1])[0, 1])**2.0
    curves = {}
    curves['Measured HU'] = {'x': x, 'y': y, 'yerr': ysd, 'dots': True,
                             'size': 5, 'color': 'r', 'beam-color': 0.0,
                             'plotText': curveLabels,
                             'plotTextAlignment': 'lowerRight'}

    curves['Linear Regression'] = {'x': np.array([x.min(), x.max()]),
                                   'y': np.array([x.min() * regressionCoef[0] +
                                                  regressionCoef[1],
                                                  x.max() * regressionCoef[0] +
                                                  regressionCoef[1]]),
                                   'size': 1, 'color': 0.0}
    dataTable = [['Material'], ['Effective [HU]'],
                 ['Measured [HU]'], ['Stddev [HU]'], ['Effective Energy'],
                 ['Reg. Coeff.']]
    for key, value in sorted(HUnom.items(), key=lambda x: x[1]):
        dataTable[0].append(key)
        dataTable[1].append(int(np.rint(HUeff[key])))
        dataTable[2].append(int(np.rint(HUmean[key])))
        dataTable[3].append(int(np.rint(HUstd[key])))
    dataTable[4].append(ekev)
    dataTable[4].append('keV')
    dataTable[5].append('ax + b')
    dataTable[5].append('a = ')
    dataTable[5].append(np.round(regressionCoef[0], 6))
    dataTable[5].append('b = ')
    dataTable[5].append(np.round(regressionCoef[1], 6))
    dataTable[5].append('R^2 = ')
    dataTable[5].append(np.round(Rsqr, 6))

    results.plots = curves
    results.plotLabels = {0: 'Measured CT-Number (HU)',
                          2: 'Effective CT-Number [' + format(ekev, 'n') +
                          ' keV] (HU)'}
    results.dataTable = dataTable

    return results


@logger
def sliceThickness(arrays, uids, pos, dz, dxdy, orientation, centerInd, ID):
    array = arrays[0]
    center = imageTools.findCP404Center(array, dxdy)

    center = np.rint(center)
    d15 = np.rint(12.5 / dxdy)
    d30 = np.rint(30.0 / dxdy)
#    d5 = np.rint(5.0 / dxdy)
    r5 = np.rint(5.0 / dxdy)

    results = Analysis("Slice Thickness", ID)
    if center[0] < 0 or center[1] < 0:
        results.success = False
        results.message = "Unable to find module center"
        return results
    labels = ['Region ' + str(i + 1) for i in range(4)]
    results.images = {'images': [array], 'pos': [pos[0]]}
    results.imageUids = [uids[0]]
    results.graphicsItems = {}

    # graphics items

    b = []
    b.append(('rect', center[1] - d30, center[0] - d30 - d15, 2 * d30, d15))
    b.append(('rect', center[1] + d30, center[0] - d30, d15,  2 * d30))
    b.append(('rect', center[1] - d30, center[0] + d30, 2 * d30, d15))
    b.append(('rect', center[1] - d30 - d15, center[0] - d30, d15,  2 * d30))
    for label, bb in zip(labels, b):
        results.graphicsItems[label] = bb
        results.graphicsItemsLabelInside[label] = False
    results.graphicsItems['Background'] = ('circle', center[1] - r5,
                                           center[0] - r5, r5 * 2, r5 * 2)

    # measurements
    q1 = array[center[0] - d30 - d15: center[0] - d30,
               center[1] - d30: center[1] + d30]
    q2 = array[center[0] - d30: center[0] + d30,
               center[1] + d30: center[1] + d30 + d15]
    q3 = array[center[0] + d30: center[0] + d30 + d15,
               center[1] - d30: center[1] + d30]
    q4 = array[center[0] - d30: center[0] + d30,
               center[1] - d30 - d15: center[1] - d30]

    bcInd = imageTools.circleIndices(array.shape, r5, center)
    bc = array[bcInd].mean()

    q = [q1, q2, q3, q4]
    FWHM = lambda b, bc: (b.max()+bc)/2.0
    p = [qq > FWHM(qq, bc) for qq in q]

    lenghts = [None] * 4

    lenghts[0] = np.count_nonzero(p[0].sum(axis=0))
    lenghts[0] *= dxdy * np.tan(np.deg2rad(23))
    lenghts[1] = np.count_nonzero(p[1].sum(axis=1))
    lenghts[1] *= dxdy * np.tan(np.deg2rad(23))
    lenghts[2] = np.count_nonzero(p[2].sum(axis=0))
    lenghts[2] *= dxdy * np.tan(np.deg2rad(23))
    lenghts[3] = np.count_nonzero(p[3].sum(axis=1))
    lenghts[3] *= dxdy * np.tan(np.deg2rad(23))

    lenghts.append(sum(lenghts) / 4.)
    lenghts.append(dz)

    header = [""] + labels
    header.append('Mean [mm]')
    header.append('Nominal [mm]')
    lenght = ['Lenght [mm]']
    lenght += [format(np.round(d, 2), ' .2f') for d in lenghts]
    devmm = ['Deviation [mm]']
    devmm += [format(np.round(d - dz, 2), ' .2f') for d in lenghts]
    devp = ['Deviation [%]']
    devp += [format(np.round(d / dz - 1., 2) * 100, ' .0f') for d in lenghts]
    devmm[-1] = ""
    devp[-1] = ""
    results.dataTable = [header, lenght, devmm, devp]
    return results


@logger
def sliceThickness3D(arrays, uids, pos, dz, dxdy, orientation, centerInd, ID):
    d15 = np.rint(12.5 / dxdy)
    d30 = np.rint(30.0 / dxdy)
    r5 = np.rint(5.0 / dxdy)

    results = Analysis("Slice Thickness 3D", ID)
    results.imageUids = uids
    results.images['images'] = []
    results.images['pos'] = []
    FWHM = lambda b, bc: (b.max()+bc)/2.0

    mean_thickness = []

    for uid, zpos, array in zip(uids, pos, arrays):
        center = imageTools.findCP404Center(array, dxdy)
        center = np.rint(center)

        center_test = (0 < center[0] < array.shape[0],
                       0 < center[1] < array.shape[1])
        if not center_test[0] or not center_test[1]:
            continue

        results.imageUids.append(uid)
        results.images['images'].append(array)
        results.images['pos'].append(zpos)

        q1 = array[center[0] - d30 - d15: center[0] - d30,
                   center[1] - d30: center[1] + d30]
        q2 = array[center[0] - d30: center[0] + d30,
                   center[1] + d30: center[1] + d30 + d15]
        q3 = array[center[0] + d30: center[0] + d30 + d15,
                   center[1] - d30: center[1] + d30]
        q4 = array[center[0] - d30: center[0] + d30,
                   center[1] - d30 - d15: center[1] - d30]

        bcInd = imageTools.circleIndices(array.shape, r5, center)
        bc = array[bcInd].mean()

        q = [q1, q2, q3, q4]

        p = [qq > FWHM(qq, bc) for qq in q]

        lenghts = np.empty(4, dtype=np.float)

        lenghts[0] = np.count_nonzero(p[0].sum(axis=0))
        lenghts[0] *= dxdy * np.tan(np.deg2rad(23))
        lenghts[1] = np.count_nonzero(p[1].sum(axis=1))
        lenghts[1] *= dxdy * np.tan(np.deg2rad(23))
        lenghts[2] = np.count_nonzero(p[2].sum(axis=0))
        lenghts[2] *= dxdy * np.tan(np.deg2rad(23))
        lenghts[3] = np.count_nonzero(p[3].sum(axis=1))
        lenghts[3] *= dxdy * np.tan(np.deg2rad(23))

        mean_thickness.append((zpos, lenghts.mean(), lenghts.std()))

    x, y, ysd = zip(*mean_thickness)
    results.plots['Measured'] = {'x': x, 'y': y, 'yerr': ysd, 'dots': True,
                                 'color': 'b', 'size': 7}

    results.plots['Nominal'] = {'x': [x[0] * 1.1, x[-1]*1.1], 'y': [dz, dz],
                                'color': 'r', 'size': 2}
    results.plotLabels[0] = 'Slice thickness [mm]'
    results.plotLabels[2] = 'Image position [mm]'
    return results


@logger
def nps(arrays, uids, pos, dz, dxdy, orientation, centerInd, ID):
    radii = [50. / dxdy, 25. / dxdy]
    angles = [np.linspace(0, 2*np.pi * (1.-1./20.), 20)]
    angles += [np.linspace(0, 2*np.pi * (1.-1./8.), 8)]
    bdia = int(np.rint(10. / dxdy))
    results = Analysis('NPS', ID)
    coords = [(0, 0)]

    # subroi coords
    for ind, r in enumerate(radii):
        for ang in angles[ind]:
            x, y = np.rint((r * np.sin(ang), r * np.cos(ang)))
            coords.append((x, y))

    # graphics
    cm = np.rint(imageTools.massCenter(arrays[0] > -300))
    for ind, c in enumerate(coords):
        results.graphicsItems[str(ind + 1)] = ('rect', cm[1] - c[1] - bdia,
                                               cm[0] - c[0] - bdia,
                                               bdia * 2, bdia * 2)
    results.images['images'] = arrays
    results.images['pos'] = pos
    results.imageUids = uids

    n = 128
    kernel_l = 5.
    kernel = np.ones(kernel_l) / kernel_l
    f, b = np.ones(kernel_l), np.ones(kernel_l)
    lineCollection = []
    for ind, arr in enumerate(arrays):
        cm = np.rint(imageTools.massCenter(arr > -300))
        lines = []
        for c in coords:
            box = np.copy(arr[cm[0]+c[0]-bdia: cm[0]+c[0]+bdia,
                              cm[1]+c[1]-bdia: cm[1]+c[1]+bdia])
            fbox = (np.abs(np.fft.fft2(box-box.mean(), s=(n, n))) /
                    np.sqrt(np.prod(box.shape))) ** 2.
            x, y = imageTools.fft2DRadProf(fbox, dxdy)

            lines.append(y)
            lineCollection.append(y)
        ps = np.array(lines).mean(0)
        yc = np.convolve(np.hstack((f * y[0], ps, b * y[-1])), kernel,
                         mode='same')[kernel_l: -kernel_l]
        fs = x * 10.
        results.plots[format(pos[ind], '.2f')] = {'y': yc, 'x': fs,
                                                  'color': (ind, len(arrays))}
    # Total NPS
    psAll = np.array(lineCollection).mean(0)
    ycAll = np.convolve(np.hstack((f * y[0], psAll, b * y[-1])), kernel,
                        mode='same')[kernel_l: -kernel_l]
    results.plots['Total'] = {'x': x * 10., 'y': ycAll, 'color': 'k',
                              'size': 2}

    results.plotLabels = {0: 'NPS', 2: 'll/cm'}
    return results


@logger
def noise(arrays, uids, pos, dz, dxdy, orientation, centerInd, ID):

    array2d = arrays[centerInd]
    array3d = np.rollaxis(np.array(arrays), 0, 3)

    cm = imageTools.massCenter(array2d > -100)
    xcm = np.rint(cm[0])
    ycm = np.rint(cm[1])
    if xcm == 0. or ycm == 0.:
        errdata = Analysis('Noise', ID)
        errdata.message = "Could not locate anything in image"
        errdata.success = False
        return errdata
    dimxy = array2d.shape[0]

    radius = np.rint(50.0 / dxdy)

    if radius >= np.array([xcm, ycm, dimxy-xcm, dimxy-ycm]).max():
        radius = np.array([xcm, ycm, dimxy-xcm, dimxy-ycm]).max() - 1

    mask2D = imageTools.circleMask((dimxy, dimxy), radius,
                                   (xcm, ycm)).reshape((dimxy, dimxy, 1))
    mask3D = np.repeat(mask2D, array3d.shape[2], axis=2)
    array_ma = np.ma.masked_where(mask3D == 0, array3d)

    noiseSlice = np.array([array_ma[:, :, k].std() for k in xrange(len(pos))])
    noiseTotal = array_ma.std()
    meanSlice = np.array([array_ma[:, :, k].mean() for k in xrange(len(pos))])
    meanTotal = array_ma.mean()

    data = Analysis('Noise', ID)
    data.imageUids = uids
    data.plotLabels[0] = 'Standard Deviation (HU)'
    data.plotLabels[2] = 'Image Position (mm)'
    if len(pos) > 1:
        data.plots['Noise'] = {'x': pos, 'y': noiseSlice, 'dots': True,
                               'color': 'b', 'size': 5}
        data.plots['Mean Noise'] = {'x': [pos[0], pos[-1]],
                                    'y': [noiseTotal] * 2,
                                    'dots': False, 'color': 'r', 'size': 3, }

    if len(pos) > 2:
        data.dataTable = [['', 'First Image', 'Center Image', 'Last Image',
                           'Total'], [], []]
        data.dataTable[1] += ['St.Dev [HU]']
        data.dataTable[1] += [np.round(noiseSlice[0], 2)]
        data.dataTable[1] += [np.round(noiseSlice[centerInd], 2)]
        data.dataTable[1] += [np.round(noiseSlice[-1], 2)]
        data.dataTable[1] += [np.round(noiseTotal, 2)]
        data.dataTable[2] += ['Mean [HU]']
        data.dataTable[2] += [np.round(meanSlice[0], 2)]
        data.dataTable[2] += [np.round(meanSlice[centerInd], 2)]
        data.dataTable[2] += [np.round(meanSlice[-1], 2)]
        data.dataTable[2] += [np.round(meanTotal, 2)]

        image = np.vstack(tuple([np.squeeze(array3d[:, :, k]) for k in [0,
                                 centerInd, -1]]))
        data.images['images'] = [image]
        data.images['pos'] = [""]

        data.graphicsItems['First'] = ('circle', ycm - radius, xcm - radius,
                                       radius * 2, radius * 2)
        data.graphicsItems['Center'] = ('circle', ycm - radius,
                                        xcm + dimxy - radius, radius * 2,
                                        radius * 2)
        data.graphicsItems['Last'] = ('circle', ycm - radius,
                                      xcm+dimxy*2 - radius, radius * 2,
                                      radius * 2)
    else:
        data.dataTable = [['', 'Center Image', 'Total'], [], []]
        data.dataTable[1] += ['St.Dev [HU]']
        data.dataTable[1] += [np.round(noiseSlice[centerInd], 2)]
        data.dataTable[1] += [np.round(noiseTotal, 2)]
        data.dataTable[2] += ['Mean [HU]']
        data.dataTable[2] += [np.round(meanSlice[centerInd], 2)]
        data.dataTable[2] += [np.round(meanTotal, 2)]

        data.images['images'] = [np.squeeze(array3d[:, :, centerInd])]
        data.images['pos'] = [""]
        data.graphicsItems['Center'] = ('circle', ycm - radius, xcm - radius,
                                        radius * 2, radius * 2)

    return data


@logger
def homogeneity(arrays, uids, pos, dz, dxdy, orientation, centerInd, ID):
    center_im = arrays[centerInd]
    xcm, ycm = imageTools.massCenter(center_im > -100.0)
    bsize = 8.0 / dxdy
    analyseRadii = 80.0 / dxdy - bsize * 2
    header = ['1', '2', '3', '4', 'Center']
    dpxy = {'1': (-1, 0), '2': (0, 1), '3': (1, 0), '4': (0, -1),
            'Center': (0, 0)}
    arr_dim = center_im.shape

    data = Analysis('Uniformity', ID)
    data.dataTable = [['', 'Region 1', 'Region 2', 'Region 3', 'Region 4',
                       'Center'],
                      ['Mean [HU]'], ['Standard Deviation [HU]'],
                      ['Deviation Center [HU]']]
    data.imageUids = uids
    data.images = {'images': [center_im], 'pos': [pos[centerInd]]}

    measure = {'pos': [], 'mean': [], 'std': []}
    # looping over images
    for ind, im, pos in zip(range(len(arrays)), arrays, pos):
        valHU = {}
        # getting mean and std from each roi
        for key, value in dpxy.items():
            x = xcm + analyseRadii * value[0]
            y = ycm + analyseRadii * value[1]
            arrInd = imageTools.circleIndices(arr_dim, bsize, (x, y))
            valHU[key] = (np.mean(im[arrInd]), np.std(im[arrInd]))
        # finding deviation between center and perifery
        dev = np.array([valHU['Center'][0]-valHU[key][0]
                        for key in header[:4]])

        measure['pos'].append(pos)
        measure['mean'].append(dev.mean())
        measure['std'].append(dev.std())
        if ind == centerInd:
            for key, pos in dpxy.items():
                data.dataTable[1].append(format(valHU[key][0], '.2f'))
                data.dataTable[2].append(format(valHU[key][1], '.2f'))
                if key != 'Center':
                    diff = valHU['Center'][0] - valHU[key][0]
                    data.dataTable[3].append(format(diff, '.2f'))

                x = xcm + analyseRadii * pos[0]
                y = ycm + analyseRadii * pos[1]
                data.graphicsItems[key] = ('circle', y - bsize, x - bsize,
                                           bsize * 2, bsize * 2)
    if len(measure['pos']) > 1:
        data.plots['Deviation Center'] = {'x': measure['pos'],
                                          'y': measure['mean'],
                                          'yerr': measure['std'],
                                          'size': 6, 'dots': True,
                                          'color': 'r'}
        data.plotLabels = {0: 'Mean deviation from center [HU]',
                           2: 'Position [mm]'}

    return data


def mtfFT2Dinterpy(f, F, mtf):
    # finding %@ mtf values
    if F.min() > mtf:
        return -1
    fp = np.argmin(np.fabs(F - mtf))
    fq = f[fp]
    Fa = F[fp]
    diff = mtf-Fa
    ii = 0
    while np.fabs(diff) > 0.01 and ii < 50:
        fq -= diff*2.
        Fa = np.interp(fq, f, F, right=0.)
        diff = mtf - Fa
        ii += 1
    return fq


@logger
def mtfFT3D(arrays, uids, pos, dz, dxdy, centerInd, orientation, ID,
            threshold=700):
    arr = np.rollaxis(np.array(arrays), 0, 3)
    cmx, cmy, cmz = np.rint(imageTools.massCenter(arr > -300))
    # testing for big enought image
    result = Analysis('MTF3D', ID)
    if 80. / dxdy > min(arr.shape[:2]):
        result.success = False
        result.message = "Could not locate Tungsten bead"
        return result
    d40 = np.rint(40. / dxdy)
    d20 = np.rint(10. / dxdy)
    d10 = np.rint(10. / dxdy)

    mtfArr = np.copy(arr[cmx - d40: cmx + d40, cmy - d20: cmy + d20, :])
    noiseArr = np.copy(arr[cmx - d10: cmx + d10, cmy - d10: cmy + d10, :])
    # correction of threshold
    thres_est = noiseArr.max() + 2 * noiseArr.std()
    threshold = max([threshold, thres_est])

    peaks = []

    peak_p = peak_local_max(mtfArr, min_distance=0, threshold_abs=threshold,
                            threshold_rel=0., exclude_border=False)

    if len(peak_p) > 0:
        for p in peak_p:
            if mtfArr[p[0], p[1], p[2]] > threshold:
                peaks.append((p[2], p[0]+cmx-d40, p[1]+cmy-d20))

    if len(peaks) == 0:
        if threshold > max([600, thres_est]):
            return mtfFT3D(arrays, uids, pos, dz, dxdy, orientation, centerInd,
                           ID, threshold=600)
        elif threshold > max([500, thres_est]):
            return mtfFT3D(arrays, uids, pos, dz, dxdy, orientation, centerInd,
                           ID, threshold=500)
        elif threshold > max([400, thres_est]):
            return mtfFT3D(arrays, uids, pos, dz, dxdy, orientation, centerInd,
                           ID, threshold=400)
        elif threshold > max([200, thres_est]):
            return mtfFT3D(arrays, uids, pos, dz, dxdy, orientation, centerInd,
                           ID, threshold=200)
        elif threshold > max([100, thres_est]):
            return mtfFT3D(arrays, uids, pos, dz, dxdy, orientation, centerInd,
                           ID, threshold=100)
        else:
            result.success = False
            result.message = "Could not locate Tungsten bead"
            return result

    # sorting peaks
    peaks_s = [[peaks.pop(0)], []]
    peaks_seed = peaks_s[0][0]
    dist = lambda p1, p2: ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 +
                           (p1[2] - p2[2])**2)**.5
    while len(peaks) > 0:
        if dist(peaks[0], peaks_seed) > d20 / 2.:
            peaks_s[1].append(peaks.pop(0))
        else:
            peaks_s[0].append(peaks.pop(0))

    beads = ['Bead 1', 'Bead 2']
    kernel_l = np.int(np.rint(.1 * d10))
    normalize = lambda mtn: (mtn - mtn[-1]) / (mtn[0] - mtn[-1])
    plots = []
    image = None
    image_pos = ""
    for ind, p_c in enumerate(peaks_s):
        if len(p_c) == 0:
            continue
        # peak position
        zind = np.sort(np.unique(np.array([x[0] for x in p_c])))
        xind = np.sort(np.unique(np.array([x[1] for x in p_c])))
        yind = np.sort(np.unique(np.array([x[2] for x in p_c])))
        # center point
        p = np.zeros(3, dtype=np.int)
        p_val = 0.
        for z in zind:
            for y in yind:
                for x in xind:
                    if arr[x, y, z] > p_val:
                        p_val = arr[x, y, z]
                        p[:] = [x, y, z]


#        z_0 = np.rint(np.median(zind)).astype(np.int)
#        p = (np.rint(np.median(xind)).astype(np.int),
#             np.rint(np.median(yind)).astype(np.int))
        # Adding images to images containing analysis list
        result.imageUids += [uids[z] for z in zind]

        # adding image and graphics to results
        result.graphicsItemsLabelInside[beads[ind]] = False
        if image is not None:
            sim = image.shape[0]
            result.graphicsItems[beads[ind]] = ('rect', p[1]-d10/2,
                                                p[0]-d10/2+sim, d10, d10)
            image = np.vstack((image, np.copy(arr[:, :, p[2]])))
            image_pos += ", " + format(pos[p[2]], '.2f')
        else:
            image = np.copy(arr[:, :, p[2]])
            result.graphicsItems[beads[ind]] = ('rect', p[1]-d10/2, p[0]-d10/2,
                                                d10, d10)
            image_pos = format(pos[p[2]], '.2f')

        # Finding MTF

        parr = np.copy(arr[p[0]-d10: p[0]+d10,
                       p[1]-d10: p[1]+d10, zind])

        masks = []
        for i in range(parr.shape[2]):
            masks.append(imageTools.circleMask(parr.shape[:2], d10 / 2,
                                               (parr.shape[0] / 2,
                                                parr.shape[1] / 2)))

        mask = np.rollaxis(np.array(masks), 0, 3)
        background = np.ma.masked_where(mask, parr)

        pool = np.mean(background)

        parr -= pool
        noise = np.std(background)

#        import pylab as plt
#        z0 = p[2] - zind[0]
#
#        plt.imshow(parr[:,:,z0], cmap='bone', vmin=-350, vmax=300, interpolation='nearest')
#        plt.savefig('E://nacp foredrag//mtf//parrHigh_uncorr')
#        plt.close('all')

#        m = np.squeeze(np.abs(np.fft.fftn(parr)[:, :, 0]))


#        k = m[:]
#        k[0,0] = k[0,1]
#
#        plt.imshow(np.fft.fftshift(k), cmap='bone',interpolation='nearest')
#        plt.savefig('E://nacp foredrag//mtf//fftHigh_uncorr')
#        plt.close('all')


        parr[parr < -noise] = -noise


#        plt.imshow(parr[:,:,z0], cmap='bone', vmin=-350, vmax=300, interpolation='nearest')
#        plt.savefig('E://nacp foredrag//mtf//parrHigh_corr')
#        plt.close('all')


#        zind_0 = p[2] - zind[0]
#        m = np.squeeze(np.abs(np.fft.fftn(parr[:, :, zind_0])))
        m = np.squeeze(np.abs(np.fft.fftn(parr)[:, :, 0]))


#        k = m[:]
#        k[0,0] = k[0,1]
#
#        plt.imshow(np.fft.fftshift(k), cmap='bone',interpolation='nearest')
#        plt.savefig('E://nacp foredrag//mtf//fftHigh_corr')
#        plt.close('all')



        x, y = imageTools.fft2DRadProf(m, dxdy)

        if kernel_l > 3:
            kernel_l = 3
        if kernel_l >= 2:
            kernel = np.ones(kernel_l) / kernel_l
            f, b = np.ones(kernel_l) * y[0], np.ones(kernel_l) * y[-1]
            yc = np.convolve(np.hstack((f, y, b)), kernel,
                             mode='same')[kernel_l: -kernel_l]
            ys = normalize(yc)
        else:
            ys = normalize(y)

        plots.append((x[:] * 10., ys[:]))

    # Cleaning and adding results

    # Adding image to result
    result.images['images'] = [image]
    result.images['pos'] = [image_pos]
    # Adding table and plots
    result.dataTable = [['MTF (ll/cm)'], ['50%'], ['10%'], ['2%']]
    if len(plots) > 1:
        colors = ['b', 'r']
        for indc, plot in enumerate(plots):
            name = beads[indc]
            freq, mtf = plot
            mtfV = [.5, .1, .02]
            mtf50, mtf10, mtf2 = [mtfFT2Dinterpy(freq, mtf, v) for v in mtfV]

            result.plots[name] = {'x': freq, 'y': mtf, 'size': 2,
                                  'color': colors[indc % len(colors)]}
            result.dataTable[0] += [name]
            result.dataTable[1] += [format(mtf50, '.4f')]
            result.dataTable[2] += [format(mtf10, '.4f')]
            result.dataTable[3] += [format(mtf2, '.4f')]
    else:
        freq, mtf = plots[0]
        mtfV = [.5, .1, .02]
        mtf50, mtf10, mtf2 = [mtfFT2Dinterpy(freq, mtf, v) for v in mtfV]

        plotText = []
        for ind, mtfp in enumerate([mtf50, mtf10, mtf2]):
            if mtfp >= 0.0:
                plotText.append((mtfp, mtfV[ind], format(mtfV[ind], '.2f')))

        result.plots['Bead'] = {'x': freq, 'y': mtf, 'size': 2, 'color': 'k'}
        if len(plotText) > 0:
            result.plots['Bead']['plotText'] = plotText
            result.plots['Bead']['plotTextAlignment'] = 'centerRight'
        result.dataTable[0] += ['Bead']
        result.dataTable[1] += [format(mtf50, '.4f')]
        result.dataTable[2] += [format(mtf10, '.4f')]
        result.dataTable[3] += [format(mtf2, '.4f')]
    result.plotLabels = {0: 'MTF (AU)', 2: 'll/cm'}
    return result


@logger
def mtf1D(arrays, uids, pos, dz, dxdy, centerInd, orientation, ID,
          threshold=700):
    arr = np.rollaxis(np.array(arrays), 0, 3)
    cmx, cmy, cmz = np.rint(imageTools.massCenter(arr > -300))
    # testing for big enought image
    result = Analysis('MTF1D(Experimental)', ID)
    result.warning = "This analysis is experimental and "
    result.warning += "results may not be reproducible"
    if 80. / dxdy > min(arr.shape[:2]):
        result.success = False
        result.message = "Could not locate Tungsten bead"
        return result
    d40 = np.rint(40. / dxdy)
    d20 = np.rint(10. / dxdy)
    d10 = np.rint(10. / dxdy)
    d5 = np.rint(5. / dxdy)

    mtfArr = np.copy(arr[cmx - d40: cmx + d40, cmy - d20: cmy + d20, :])
    noiseArr = np.copy(arr[cmx - d10: cmx + d10, cmy - d10: cmy + d10, :])
    # correction of threshold
    thres_est = noiseArr.max() + 2 * noiseArr.std()
    threshold = max([threshold, thres_est])

    peaks = []
    peak_p = peak_local_max(mtfArr, min_distance=0, threshold_abs=threshold,
                            threshold_rel=0., exclude_border=False)
    if len(peak_p) > 0:
        for p in peak_p:
            if mtfArr[p[0], p[1], p[2]] > threshold:
                peaks.append((p[2], p[0]+cmx-d40, p[1]+cmy-d20))

    if len(peaks) == 0:
        if threshold > max([600, thres_est]):
            return mtfFT3D(arrays, uids, pos, dz, dxdy, orientation, centerInd,
                           ID, threshold=600)
        elif threshold > max([500, thres_est]):
            return mtfFT3D(arrays, uids, pos, dz, dxdy, orientation, centerInd,
                           ID, threshold=500)
        elif threshold > max([400, thres_est]):
            return mtfFT3D(arrays, uids, pos, dz, dxdy, orientation, centerInd,
                           ID, threshold=400)
        elif threshold > max([200, thres_est]):
            return mtfFT3D(arrays, uids, pos, dz, dxdy, orientation, centerInd,
                           ID, threshold=200)
        elif threshold > max([100, thres_est]):
            return mtfFT3D(arrays, uids, pos, dz, dxdy, orientation, centerInd,
                           ID, threshold=100)
        else:
            result.success = False
            result.message = "Could not locate Tungsten bead"
            return result

    # sorting peaks
    peaks_s = [[peaks.pop(0)], []]
    peaks_seed = peaks_s[0][0]
    dist = lambda p1, p2: ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 +
                           (p1[2] - p2[2])**2)**.5
    while len(peaks) > 0:
        if dist(peaks[0], peaks_seed) > d20 / 2.:
            peaks_s[1].append(peaks.pop(0))
        else:
            peaks_s[0].append(peaks.pop(0))

    beads = ['Bead 1', 'Bead 2']
    kernel_l = np.int(np.rint(.1 * d10))
    normalize = lambda mtn: (mtn - mtn[-1]) / (mtn[0] - mtn[-1])
    plots = []
    image = None
    image_pos = ""
    for ind, p_c in enumerate(peaks_s):
        if len(p_c) == 0:
            continue
        # peak position
        zind = np.sort(np.unique(np.array([x[0] for x in p_c])))
        xind = np.sort(np.unique(np.array([x[1] for x in p_c])))
        yind = np.sort(np.unique(np.array([x[2] for x in p_c])))

        # center point
        p = np.zeros(3, dtype=np.int)
        p_val = 0.
        for z in zind:
            for y in yind:
                for x in xind:
                    if arr[x, y, z] > p_val:
                        p_val = arr[x, y, z]
                        p[:] = [x, y, z]

        # Adding images to images containing analysis list
        result.imageUids += [uids[z] for z in zind]

        # adding image and graphics to results
        result.graphicsItemsLabelInside[beads[ind]] = False
        if image is not None:
            sim = image.shape[0]
            result.graphicsItems[beads[ind]] = ('rect', p[1]-d10/2,
                                                p[0]-d10/2+sim, d10, d10)
            image = np.vstack((image, np.copy(arr[:, :, p[2]])))
            image_pos += ", " + format(pos[p[2]], '.2f')
        else:
            image = np.copy(arr[:, :, p[2]])
            result.graphicsItems[beads[ind]] = ('rect', p[1]-d10/2, p[0]-d10/2,
                                                d10, d10)
            image_pos = format(pos[p[2]], '.2f')

        # Finding MTF
        parr = np.squeeze(np.copy(arr[p[0]-d10: p[0]+d10,
                                      p[1]-d10 + 1: p[1]+d10 + 1, p[2]]))

        mask = imageTools.circleMask(parr.shape, d10/2)
        background = np.ma.masked_where(mask, parr)

        pool = np.mean(background)
        parr -= pool
        noise = np.std(background)
        parr[parr < -noise] = -noise

        center = np.unravel_index(np.argmax(parr), parr.shape)
        l1 = parr[center[0], center[1] - d5: center[1] + d5].ravel()
        l2 = parr[center[0] - d5: center[0] + d5, center[1]].ravel()

        fft = np.abs(np.fft.fft(l1, 128)) + np.abs(np.fft.fft(l2, 128))

        freq = np.fft.fftfreq(len(fft), dxdy)
        x = freq[freq >= 0]
        y = fft[freq >= 0]

        if kernel_l > 3:
            kernel_l = 3
        if kernel_l >= 2:
            kernel = np.ones(kernel_l) / kernel_l
            f, b = np.ones(kernel_l) * y[0], np.ones(kernel_l) * y[-1]
            yc = np.convolve(np.hstack((f, y, b)), kernel,
                             mode='same')[kernel_l: -kernel_l]
            ys = normalize(yc)
        else:
            ys = normalize(y)
        plots.append((x[:] * 10., ys[:]))

    # Cleaning and adding results

    # Adding image to result
    result.images['images'] = [image]
    result.images['pos'] = [image_pos]
    # Adding table and plots
    result.dataTable = [['MTF (ll/cm)'], ['50%'], ['10%'], ['2%']]
    if len(plots) > 1:
        colors = ['b', 'r']
        for indc, plot in enumerate(plots):
            name = beads[indc]
            freq, mtf = plot
            mtfV = [.5, .1, .02]
            mtf50, mtf10, mtf2 = [mtfFT2Dinterpy(freq, mtf, v) for v in mtfV]

            result.plots[name] = {'x': freq, 'y': mtf, 'size': 2,
                                  'color': colors[indc % len(colors)]}
            result.dataTable[0] += [name]
            result.dataTable[1] += [format(mtf50, '.4f')]
            result.dataTable[2] += [format(mtf10, '.4f')]
            result.dataTable[3] += [format(mtf2, '.4f')]
    else:
        freq, mtf = plots[0]
        mtfV = [.5, .1, .02]
        mtf50, mtf10, mtf2 = [mtfFT2Dinterpy(freq, mtf, v) for v in mtfV]

        plotText = []
        for ind, mtfp in enumerate([mtf50, mtf10, mtf2]):
            if mtfp >= 0.0:
                plotText.append((mtfp, mtfV[ind], format(mtfV[ind], '.2f')))

        result.plots['Bead'] = {'x': freq, 'y': mtf, 'size': 2, 'color': 'k'}
        if len(plotText) > 0:
            result.plots['Bead']['plotText'] = plotText
            result.plots['Bead']['plotTextAlignment'] = 'centerRight'
        result.dataTable[0] += ['Bead']
        result.dataTable[1] += [format(mtf50, '.4f')]
        result.dataTable[2] += [format(mtf10, '.4f')]
        result.dataTable[3] += [format(mtf2, '.4f')]
    result.plotLabels = {0: 'MTF (AU)', 2: 'll/cm'}
    return result


@logger
def lcd(arrays, uids, pos, dz, dxdy, orientation, centerInd, ID):
    """Low Contrast Detectability, recipy from:
    Automated assessment of low contrast sensitivity for CT systems using a
    model observer
    I. Hernandez-Giron, J. Geleijns, A. Calzado, and W. J. H. Veldkamp
    Citation: Medical Physics 38, S25 (2011),  doi: 10.1118/1.3577757
    """

    array = arrays[centerInd]

    # finding center of phantom
    cmx, cmy = imageTools.massCenter(array > -300)

    # finding angle correction
    arr3d = np.rollaxis(np.array(arrays), 0, 3)
    if np.any(arr3d >= 100):
        cmpx, cmpy, cmpz = imageTools.massCenter(arr3d >= 100)
        angle_corr = np.tan((cmy - cmpy) / (cmx - cmpx))
    else:
        angle_corr = 0.0

    # mask of low contrast objects:
    circle_diameter = np.array([15, 9, 8, 7, 6, 5, 4, 3, 2]).astype(np.float)

    circle_position = np.deg2rad(np.array([30., 150., 270.]) - 177.5)
#    circle_raddist = np.deg2rad(np.array([0., 11.25, 22.5, 33.75, 45., 56.25,
#                                          67.5, 78.75, 90.]))
    circle_raddist = np.deg2rad(np.array([0.0, 19.3066913,
                                          35.27558584, 48.6659367,
                                          62.16713806, 74.44872337,
                                          83.08319128, 90.02867232,
                                          96.10304115]))

    # applying angle correction
    circle_raddist -= angle_corr
    circle_radius = 50. / dxdy  # 50mm

    result = Analysis('LCD(Experimental)', ID)
    result.warning = "This module is experimental "
    result.warning += "and have reproducibility issues."

    # descibing low contrast rois as [(posz, posy, diameter), ...]
    rois = dict([('px', []), ('py', []), ('diameter', []), ('angle', []),
                 ('mean', []), ('std', []), ('label', []), ('contrast', []),
                 ('diameterMM', [])])

    # populating roi data
    teller = 1
    contrastLevels = [0.3, 0.5, 1.0]
    for contrast, cpos in zip(contrastLevels, circle_position):
        for ind, dpos in enumerate(circle_raddist):
            if orientation[0] < 0.0:
                py = cmy - circle_radius * np.cos(cpos + dpos)
            else:
                py = cmy + circle_radius * np.cos(cpos + dpos)
            px = cmx + circle_radius * np.sin(cpos + dpos)
            rois['px'].append(px)
            rois['py'].append(py)
            dia = circle_diameter[ind] / dxdy
            rois['diameter'].append(dia)
            rois['diameterMM'].append(circle_diameter[ind])
            rois['angle'].append(cpos + dpos)
            dia_h = dia / 2.
            roi_ind = imageTools.circleIndices(array.shape, dia_h, (px, py))
            masked = array[roi_ind]
            rois['mean'].append(masked.mean())
            rois['contrast'].append(contrast)
            rois['std'].append(masked.std())
            clabel = str(teller)
            teller += 1
            rois['label'].append(clabel)
            result.graphicsItems[clabel] = ('circle', py - dia_h, px - dia_h,
                                            dia, dia)

    # finding background
    bg_rad2 = 60. / dxdy
    bg_rad1 = 70. / dxdy
    mask = imageTools.circleMask(array.shape, bg_rad2, (cmx, cmy))
    mask -= imageTools.circleMask(array.shape, bg_rad1, (cmx, cmy))
    bg_array = np.ma.masked_where(mask == 0, array)
    bg_mean = bg_array.mean()
    bg_std = bg_array.std()
    result.graphicsItems[' '] = ('circle', cmy - bg_rad1, cmx - bg_rad1,
                                 bg_rad1 * 2., bg_rad1 * 2)
    result.graphicsItems['Background'] = ('circle', cmy - bg_rad2,
                                          cmx - bg_rad2,
                                          bg_rad2 * 2., bg_rad2 * 2)
    result.graphicsItemsLabelInside['Background'] = False

    # calculating detectability
    roi_m = np.array(rois['mean'])
    roi_std = np.array(rois['std'])
    d_index = np.abs(roi_m - bg_mean) \
        / (.5 * bg_std**2. + .5 * roi_std**2.)**.5
    PC = [0.5 + 0.5 * math.erf(d) for d in d_index]

    result.dataTable = [['Contrast', 'Detectable diameter [mm]']]

    # plotting data
    PCa = np.array(PC)
    dSizea = np.array(rois['diameterMM'])
    ca = np.array(rois['contrast'])
    inda = np.array([int(x) for x in rois['label']])
    for color, c in zip(['r', 'g', 'b'], contrastLevels):
        ind = np.where(ca == c)

        plotText = zip(list(dSizea[ind]), list(PCa[ind]),
                       [str(x) for x in inda[ind]])
        plotTitle = format(c, '.1f') + '% contrast'

        result.plots[plotTitle] = {'x': dSizea[ind], 'y': PCa[ind],
                                   'size': 6, 'dots': True, 'color': color,
                                   'plotText': plotText,
                                   'plotTextAlignment': 'lowerRight'}
        # fitting model
        succ, lam, dm, PCm = fitLowContrastScore(PCa[ind], dSizea[ind])

        if succ:
            if lam < 2.:
                result.dataTable.append([plotTitle, '< 2.0'])
            elif lam > 15.:
                result.dataTable.append([plotTitle, '> 15.0'])
            else:
                result.dataTable.append([plotTitle, format(np.round(lam, 1),
                                         '.1f')])
            fitTitle = format(c, '.1f') + '% contrast Fit'
            result.plots[fitTitle] = {'x': dm, 'y': PCm, 'size': 2,
                                      'color': color, 'inLegend': False}
        else:
            result.dataTable.append([plotTitle, 'Could not fit model'])

    result.images = {'images': [array], 'pos': [pos[centerInd]]}
    result.imageUids = [uids[centerInd]]
    result.plotLabels[0] = 'Estimated Area Under ROC curve [A.U]'
    result.plotLabels[2] = 'Target diameter [mm]'
    result.plotLegendPosition = 'upperRight'
    return result
