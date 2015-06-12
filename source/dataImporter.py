"""
Created on Wed Nov 20 10:54:48 2013

@author: erlean
"""
from __future__ import unicode_literals
from PyQt4 import QtCore
import numpy as np
import dicom
import os
from multiprocessing import Pool, cpu_count, Queue, Process
from Queue import Empty, Full
import imageTools
from qtutils import qStringToStr
import sys
import itertools
Locale = sys.getdefaultencoding()

# uses multiprocessing if more than this files
many_file_threshold = 5

thumbnailSize = 16

MULTI_PRO = True

validTags = {'patientID': (0x10, 0x20),  # patient ID
             'studyID': (0x20, 0xD),  # study id
             'seriesID': (0x20, 0xE),  # seriesID
             'imageID': (0x8, 0x18),  # imageID
             'imageOrientation': (0x20, 0x37),  # image orientation
             'imageLocation': (0x20, 0x32),  # image location
             'patientPosition': (0x18, 0x5100),  # patient position
             'pixelSpacing': (0x28, 0x30),  # pixel spacing
             'sliceSpacing': (0x18, 0x50),  # slice spacing
             'aqusitionNumber': (0x20, 0x12),  # acu number
             'coordID': (0x20, 0x52),  # FOR ID
             }
extraTags = {'seriesDate': (0x8, 0x22),
             'seriesTime': (0x8, 0x31),
             'manufacturer': (0x8, 0x70),
             'modelName': (0x8, 0x1090),
             'kvp': (0x18, 0x60),
             'sFOV': (0x18, 0x90),
             'protocol': (0x18, 0x1030),
             'rFOV': (0x18, 0x1100),
             'exposure': (0x18, 0x1151),
             'exposuretime': (0x18, 0x1150),
             'mas': (0x18, 0x1152),
             'focus': (0x18, 0x1190),
             'kernel': (0x18, 0x1210),
             'ctdiVol': (0x18, 0x9345),
             'seriesNumber': (0x20, 0x11),
             'seriesDescription': (0x8, 0x103e),
             'institutionName': (0x8, 0x80),
             'siteName': (0x8, 0x1010),
             'software': (0x18, 0x1020),
             'bodypart': (0x18, 0x15),
             'filter': (0x18, 0x1160),
             'singlecoll': (0x18, 0x9306),
             'multicoll': (0x18, 0x9307),
             'pitch': (0x18, 0x9311),
             'rotTime': (0x18, 0x9305),
             }


def dicom_type_translator(tag):
    vm = dicom.datadict.dictionaryVM(tag)
    if vm != '1':
        return np.array
    tp = dicom.datadict.dictionaryVR(tag)
    if tp in ['LO', 'DA', 'SH', 'UI', 'CS', 'TM']:
        return str
    elif tp in ['IS']:
        return int
    elif tp in ['DS', 'FD']:
        return float
    else:
        return str


def centerCP404(im, center, pixel_spacing, orientation):
    """
    Purpose: Attemting to locate center of CP404 CatPhan module by locating the
    four aluminium ramps used for slice thickness calculations.
    Returns the average deviation from center of phantom for each ramp
    """

    array = (im > 150).astype(np.float)
#    array = im
    center = np.rint(center)
    d15 = np.rint(12.5 / pixel_spacing)
    d30 = np.rint(30.0 / pixel_spacing)
    d5 = np.rint(5.0 / pixel_spacing)

    q1 = im[center[0] - d30 - d15: center[0] - d30,
            center[1] - d30: center[1] + d30]
    q2 = im[center[0] - d30: center[0] + d30,
            center[1] + d30: center[1] + d30 + d15]
    q3 = im[center[0] + d30: center[0] + d30 + d15,
            center[1] - d30: center[1] + d30]
    q4 = im[center[0] - d30: center[0] + d30,
            center[1] - d30 - d15: center[1] - d30]
    bc = np.mean(array[center[0] - d5: center[0] + d5,
                       center[1] - d5: center[1] + d5])
    q = [q1, q2, q3, q4]
    FWHM = lambda b, bc: (b.max()+bc)/2.0
    p = [(qq > FWHM(qq, bc)).sum(0) > 0 for qq in q if all(qq.shape)]

    dev = []
    for i, r in enumerate(p):
        if r.sum() > 0:
            k = 1 - 2 * (i > 1)
            dev.append(k*np.sum(r * np.arange(r.shape[0])) / r.sum()
                       - k * r.shape[0] / 2.)

    if len(dev) > 0:
        deviation = sum(dev) / float(len(dev)) \
            * np.tan(np.deg2rad(23.)) * pixel_spacing * 2.
        deviation *= orientation
        success = True
    else:
        success = False
        deviation = 0
    return success, deviation


def imageInCP404(image, pixel_spacing):

    x, y = imageTools.findCP404Rods(image, pixel_spacing)
    if len(x) != 4:
        return False, (-1, -1)
    return True, (x.mean(), y.mean())

#    #Area test
#    A = 0.
#    Cx = 0.
#    Cy = 0.
#    ind = np.hstack((np.arange(len(x)), 0))
#
#    for i in range(len(x)):
#        m = x[ind[i]] * y[ind[i+1]] - x[ind[i+1]] * y[ind[i]]
#        A += m
#        Cx += (x[ind[i]] + x[ind[i+1]]) * m
#        Cy += (y[ind[i]] + y[ind[i+1]]) * m
#    A /= 2.
#    Cx /= 6. * A
#    Cy /= 6. * A
#    A = np.abs(A)
#    rod_spacing = imageTools.cp404_rodSpacing / float(pixel_spacing)
#    # Mean deviation in calc rod lenght: dA = (x+metric)**2 - x**2
#    metric = np.abs(A - rod_spacing ** 2) / float((2 * rod_spacing + 1))
#
#    if metric * float(pixel_spacing) > 5.:
#        return False, (-1, -1)
#    else:
#        return True, (Cx, Cy)


def validImage(path):
    data = {}
    success = True
    try:
        dc = dicom.read_file(qStringToStr(path))
    except dicom.filereader.InvalidDicomError:
        data['message'] = 'Not DiCOM'
        success = False
    else:
        for desk, tag in validTags.items():
            try:
                f = dicom_type_translator(tag)
                data[tag] = f(dc[tag].value)
            except KeyError:
                ht = '(' + format(tag[0], 'X') + ',' + format(tag[1], 'X')
                ht += ')'
                data['message'] = 'Invalid DiCOM, Could not find tag ' + ht
                success = False
            except ValueError:
                if tag is validTags['aqusitionNumber']:
                    data[validTags['aqusitionNumber']] = 1
                else:
                    ht = '(' + format(tag[0], 'X') + ',' + format(tag[1], 'X')
                    ht += ')'
                    data['message'] = 'Invalid DiCOM, Error in tag ' + ht
                    success = False
            except TypeError:
                if tag is validTags['aqusitionNumber']:
                    data[validTags['aqusitionNumber']] = 1
                else:
                    ht = '(' + format(tag[0], 'X') + ',' + format(tag[1], 'X')
                    ht += ')'
                    data['message'] = 'Invalid DiCOM, data type error in tag '
                    data['message'] += ht
                    success = False
        if success:
            # Test for orientation
            orient = dc[validTags['imageOrientation']].value
            xvec, yvec = np.array(orient[:3]), np.array(orient[3:])
            zvec = np.cross(xvec, yvec)
            if abs(zvec[2]) < 0.99:
                data['message'] = 'Image not Axial'
                success = False
#            # test for valid aquisition number
#            try:
#                int(data[validTags['aqusitionNumber']])
#            except ValueError:
#                data[validTags['aqusitionNumber']] = 1
#            else:
#                data[validTags['aqusitionNumber']] = int(data[validTags[
#                    'aqusitionNumber']])

    if success:
        image = imageTools.pixelArray(dc)
        pixel_spacing = float(data[validTags['pixelSpacing']][0])
        # extracting thumbnail
        imRed = imageTools.rebin(image, (thumbnailSize, thumbnailSize))
        data['thumbnail'] = imRed

        # looking for CP404 module
        try:
            metric, center = imageInCP404(image,
                                          pixel_spacing)
        except Exception, e:
            # todo: log exception
            metric, center = (False, (-1, -1))

        data['center'] = center
        if metric:
            # Finding image direction and location
            dir_vec = np.array(dc[validTags['imageOrientation']].value)
            orientation = np.cross(dir_vec[:3], dir_vec[3:])[2]
            pos_success, phantom_pos = centerCP404(image, center,
                                                   pixel_spacing, orientation)
            if pos_success:
                data['phantomPos'] = phantom_pos
                data['inCP404'] = True
            else:
                data['phantomPos'] = None
                data['inCP404'] = False
        else:
            data['phantomPos'] = None
            data['inCP404'] = False
        # finding extra data
        for desk, tag, in extraTags.items():
            try:
                data[tag] = dc[tag].value
            except KeyError:
                pass

    return success, (path, data)


def findAllFilesQt(pathList):
    for qstpath in pathList:
        info = QtCore.QFileInfo(qstpath)
        if info.isFile():
            yield info.canonicalFilePath()
        elif info.isDir():
            dd = QtCore.QDir(info.canonicalFilePath())
            it = QtCore.QDirIterator(dd, QtCore.QDirIterator.Subdirectories)
            while it.hasNext():
                it.next()
                fileInfo = it.fileInfo()
                if fileInfo.isFile():
                    yield fileInfo.canonicalFilePath()


def findAllFiles(pathList):
    for path in pathList:
        if os.path.isdir(path):
            for dirname, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    yield os.path.normpath(os.path.join(dirname, filename))
        elif os.path.isfile(path):
            yield os.path.normpath(path)


def data_importerMP(in_q, out_q):
    while True:
        p = in_q.get()
        if p != 'STOP':
            r = validImage(p)
            out_q.put(r)
        else:
            break


class DataImporter(QtCore.QObject):
    progress = QtCore.pyqtSignal(int)
    progressInit = QtCore.pyqtSignal(int, int)
    imageFound = QtCore.pyqtSignal(tuple)
    logMessage = QtCore.pyqtSignal(QtCore.QString)
    importFinished = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super(DataImporter, self).__init__(parent)
        self.input_queue = Queue()
        self.output_queue = Queue()

    @QtCore.pyqtSlot(list)
    def importPath(self, urlList):

        # listing available files
        self.progressInit.emit(0, 0)

        for url in urlList:
            dbtxt = QtCore.QString("Importing: ") + url.toLocalFile()
            self.logMessage.emit(dbtxt)

#        os.walk for using os.walk to walking a dir tree
#        pathList = [str(url.toLocalFile()) for url in urlList]
#        filejob = pool.map(findAllFiles, pathList)
#        files = []
#        for filelist in filejob:
#            files += filelist

# QT For using QT for walking a directory tree
        strList = [url.toLocalFile() for url in urlList]
        filejob = findAllFilesQt(strList)

#        files = []
#        for filelist in filejob:
#            files += filelist


#        teller = 1
#        self.progressInit.emit(0, nFiles)
#        self.progress.emit(1)
        if MULTI_PRO:
            ncpu = cpu_count()
            if ncpu < 1:
                ncpu = 1

            processes = []
            for _ in range(ncpu):

                processes.append(Process(target=data_importerMP,
                                         args=(self.input_queue,
                                               self.output_queue)))
            for p in processes:
                p.daemon = True
                p.start()

            nFiles = 0
            for path in filejob:
                self.input_queue.put(path)
                nFiles += 1

            dbtxt = QtCore.QString("Found: " + str(nFiles) + " files:")
            self.logMessage.emit(dbtxt)
#            self.progress.emit(1)
            for _ in processes:
                self.input_queue.put('STOP')
            self.progressInit.emit(0, nFiles)
            for ind in range(nFiles):
                job = self.output_queue.get()
                success, info = job
                path, data = info
                if success:
                    self.imageFound.emit(info)
                    log = QtCore.QString("Imported ") + path
                else:
                    log = QtCore.QString("Not imported ")
                    log += path + QtCore.QString(": ")
                    log += QtCore.QString(data['message'])

                self.logMessage.emit(log)
                self.progress.emit(ind + 1)

            for p in processes:
                p.join()
        else:

            files = list(filejob)
            self.progressInit.emit(0, len(files))
            for ind, path_1 in enumerate(files):
                success, info = validImage(path_1)
                path, data = info
                if success:
                    self.imageFound.emit(info)
                    log = QtCore.QString("Imported ") + path
                else:
                    log = QtCore.QString("Not imported ")
                    log += path + QtCore.QString(": ")
                    log += QtCore.QString(data['message'])

                self.logMessage.emit(log)
                self.progress.emit(ind + 1)

        self.progress.emit(ind + 2)
        self.progressInit.emit(0, 1)
        self.progress.emit(3)
        self.importFinished.emit()
