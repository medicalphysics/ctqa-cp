"""
Created on Fri Dec 06 11:03:45 2013

@author: erlean
"""

from PyQt4 import QtCore
from dataManager import Root
import numpy as np


class Localizer(QtCore.QObject):
    positionCorrection = QtCore.pyqtSignal(list, float, bool)
    runningStatus = QtCore.pyqtSignal(bool)
    progress = QtCore.pyqtSignal(int)
    progressInit = QtCore.pyqtSignal(int, int)

    def __init__(self, parent=None):
        super(Localizer, self).__init__(parent)

    def validCookDistance(self, x, y):
        a, b, sumsqr = self.linreg(x, y)

        if np.isnan(sumsqr):
            return np.ones(len(x)).astype(np.bool)
#        ##correction for intel MKL error in DGELSD
#        if len(sumsqr) == 0:
#            sumsqr = 1.
        xa = np.array(x).astype(np.float)
        ya = np.array(y).astype(np.float)
        Yj = xa * a + b
        mse = sumsqr / float(len(x))
        X = np.asmatrix(np.vstack((xa, np.ones(len(x)))).T)
        H = np.asarray(np.diag(X * (X.T * X)**(-1) * X.T))

        # cook distance
        C = (ya - Yj)**2 * H / (2 * mse * (1. - H)**2)
        return C <= 4. / float(len(x))

    def linreg(self, x, y):
        xa = np.vstack((x.astype(np.float), np.ones(len(x)))).T
        ya = np.array(y).astype(np.float)
        try:
            par, res = np.linalg.lstsq(xa, ya)[:2]
        except np.linalg.LinAlgError:
            par = (-1, -1)
            res = np.nan
#            print 'Errpr'
        return par[0], par[1], res

    @QtCore.pyqtSlot(Root)
    def localize(self, root):
        self.runningStatus.emit(True)
        FORs = {}
        # Finding frames of references (FORs)
        self.progressInit.emit(0, len(root.children))
        for series in root.children:
            FOR = series.imageLocation.ID
            if FOR in FORs:
                FORs[FOR].append(series)
            else:
                FORs[FOR] = [series]
        #  Finding all images in a FOR that is estimated to contain images of
        #  the CP404 module
        teller = 1
        for key, series in FORs.items():
            x = []
            y = []
            for serie in series:
                teller += 1
                self.progress.emit(teller)
                for image in serie.imageContainer.children:
                    if image.inCP404:
                        y.append(image.phantomPos)
                        x.append(image.pos)

            # apply Cook correction to filter outliers
            if len(x) >= 4:
                cind = self.validCookDistance(np.array(x), np.array(y))
            else:
                cind = np.ones(len(x)).astype(np.bool)

            if cind.sum() >= 2:
                cx = np.array(x)[cind]
                cy = np.array(y)[cind]
            else:
                cx = np.array(x)
                cy = np.array(y)

            #  if number of images in a FOR is greater than 2, do a linear
            #  regression to estimate position in phantom.
            if len(cx) > 2:
                a, b, sumsqr = self.linreg(cx, cy)
                im_orientation = series[0].patientOrientation[:2] == 'HF'
                if np.isnan(sumsqr):
                    self.positionCorrection.emit(series, float('NaN'),
                                                 im_orientation)
                else:
                    try:
                        -b / a
                    except ZeroDivisionError:
                        pass
                    else:
                        self.positionCorrection.emit(series,
                                                     np.round(-b / a, 2),
                                                     im_orientation)

            # If only two images in FOR, estimate position by a simple line
            # between positions
            elif len(cx) == 2:
                y1, y2 = cy[0], cy[1]
                x1, x2 = cx[0], cx[1]
                try:
                    a = (y2 - y1) / (x2 - x1)
                    b = y1 - a * x1
                    -b / a
                except ZeroDivisionError:
                    pass
                else:
                    im_orientation = series[0].patientOrientation[:2] == 'HF'
                    self.positionCorrection.emit(series, np.round(-b / a, 2),
                                                 im_orientation)


            # If only one image in FOR set phantom position to slice position
            # of that image. NOTE: We can not estimate image stack direction
            # and currently guessig HFS orientation
            elif len(cx) == 1:
                im_orientation = series[0].patientOrientation[:2] == 'HF'
                self.positionCorrection.emit(series, cx[0], im_orientation)
            else:
                im_orientation = series[0].patientOrientation[:2] == 'HF'
                self.positionCorrection.emit(series, float('NaN'),
                                             im_orientation)
        self.progress.emit(teller + 1)
        self.runningStatus.emit(False)
