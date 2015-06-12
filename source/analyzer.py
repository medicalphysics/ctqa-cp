"""
Created on Mon Dec 30 12:38:19 2013

@author: erlean
"""
from __future__ import unicode_literals
from PyQt4 import QtCore
from multiprocessing import Pool, cpu_count, Queue, Process
from Queue import Empty, Full
from dataManager import Root, Series
import dicom
import itertools
from qtutils import qStringToStr
import numpy as np
from dataImporter import validTags
from analysisMethods import pixelSize, HUlinarity, sliceThickness, noise
from analysisMethods import homogeneity, nps, mtfFT3D, lcd, mtf1D
import imageTools


MULTI_PRO = True  # multiprocessing on/off, for debug mostly


ascii_errorMessage = "Multiprocessing disabled due to non ascii symbols in "
ascii_errorMessage += "some file path(s). Analysis will be done on one "
ascii_errorMessage += "processor core only.\nThis is considered a bug."


# CatPhan phantom definitions
PhantomPosition = {'CatPhan600': {401: 0.0, 528: 70., 486: 160., 515: 110., },
                   'CatPhan504': {401: 0.0, 528: -30.0, 486: 80, 515:  70.}
                   }


PhantomLimits = {401: 10., 528: 15., 486: 20., 515: 10, }


# analysis methods available and phantom limits to be included for the analysis
def analysisMethods(phantomPosition):
    m = {401: [(pixelSize, 0., 0.),
               (HUlinarity, 0., 0.),
               (sliceThickness, 0., 0.)],
         486: [(noise,
                phantomPosition[486] - 15.,
                phantomPosition[486] + 15.
                ),
               (homogeneity,
                phantomPosition[486] - 10,
                phantomPosition[486] + 10
                ),
               (nps,
                phantomPosition[486] - 10.,
                phantomPosition[486] + 10.)
               ],
         528: [(mtfFT3D,
                phantomPosition[528] - 15.,
                phantomPosition[528] + 15.
                ),
#               (mtf1D,
#                phantomPosition[528] - 15.,
#                phantomPosition[528] + 15.
#                ),
               ],
         515: [(lcd,
                phantomPosition[515] - 10.,
                phantomPosition[515] + 10.
                ),
               ],
         }
    return m


def analysisArgsSetup(paths):
    dcL = [dicom.read_file(qStringToStr(path))
           for path in paths]
    array = [imageTools.pixelArray(dc) for dc in dcL]
    uids = [str(dc[validTags['imageID']].value) for dc in dcL]
    pos = [float(dc[validTags['imageLocation']].value[2]) for dc in dcL]
    dz = float(dcL[0][validTags['sliceSpacing']].value)
    dxdy = float(dcL[0][validTags['pixelSpacing']].value[0])
    orientation = np.array(dcL[0][validTags['imageOrientation']].value)
    return array, uids, pos, dz, dxdy, orientation


def analysisSetUp(seriesData):
    """ Returns a list of functions and args """
    # finding offset
    offset = seriesData['offset']
    # getting positions and images

    loc = zip(seriesData['zloc'], seriesData['paths'])
    loc.sort(key=lambda x: x[0])

    AP = seriesData['AP']

    phantomPosition = seriesData['phantomDefinition']
    methods = seriesData['analysisMethods']
    # Finding images to be analysed
    validImages = {}
    for key in phantomPosition.keys():
        ulim = phantomPosition[key] + PhantomLimits[key]
        llim = phantomPosition[key] - PhantomLimits[key]
        if AP:
            ulim, llim = -llim, -ulim
        for pos, path in loc:
            rpos = pos - offset
            if llim <= rpos <= ulim:
                if key not in validImages:
                    validImages[key] = [(path, rpos)]
                else:
                    validImages[key].append((path, rpos))

    ret = []

    # Constructing analysis
    for key, analysis in methods.items():
        for value in analysis:
            if AP:
                lAnalysisLim = -value[2]
                uAnalysisLim = -value[1]
            else:
                lAnalysisLim = value[1]
                uAnalysisLim = value[2]
            if key in validImages:
                paths, rpos = zip(*validImages[key])
                analysisPaths = []
                minp = []
                for ind, p in enumerate(rpos):
                    if lAnalysisLim <= p <= uAnalysisLim:
                        analysisPaths.append(paths[ind])
                        minp.append(np.abs(p -
                                    0.5 * (lAnalysisLim + uAnalysisLim)))
                if len(minp) == 0:
                    center = 0.5 * (lAnalysisLim + uAnalysisLim)
                    centerInd = np.argmin(np.abs(np.array(rpos) - center))
                else:
                    centerInd = np.argmin(np.array(minp))
                if len(analysisPaths) == 0:
                    analysisPaths.append(paths[centerInd])
                    centerInd = 0

                args = analysisArgsSetup(analysisPaths) + (centerInd, )
                args += (seriesData['ID'], )
                ret.append((value[0], args))
    return ret


def iter_queue(q, block=True, timeout=None, stop='STOP', stop_count=1):
    s = 0
    while True:
        try:
            r = q.get(block, timeout)
        except Empty:
            return
        else:
            if r != stop:
                yield r
            else:
                s += 1
                if s >= stop_count:
                    return


def analyse_MP(setup_queue, data_queue, result_queue, stop_queue):
    has_setup = True
    has_ana = True
    stop = False
    while True:
        try:
            ana = data_queue.get(True, 1.)
        except Empty:
            if not has_setup:
                has_ana = False
        else:
            has_ana = True
            func, args = ana
            r = apply(func, args)
            result_queue.put(r)

        try:
            setup = setup_queue.get(False)
        except Empty:
            has_setup = False
        else:
            has_setup = True

            for ana in analysisSetUp(setup):
                data_queue.put(ana)
        if not stop:
            try:
                stop_queue.get(False)
            except Empty:
                pass
            else:
                stop = True

        if stop and not has_ana and not has_setup:
            result_queue.put('STOP')
            return


class Analyzer(QtCore.QObject):
    analysis = QtCore.pyqtSignal(str, int, object)
    runningStatus = QtCore.pyqtSignal(bool)
    progress = QtCore.pyqtSignal(int)
    progressInit = QtCore.pyqtSignal(int, int)
    logMessage = QtCore.pyqtSignal(QtCore.QString)
    lock = QtCore.QMutex()

    def __init__(self, parent=None):
        super(Analyzer, self).__init__(parent)
        self.data_queue = Queue()
        self.stop_queue = Queue()
        self.setup_queue = Queue()
        self.result_queue = Queue()
        self.processes = []


    @QtCore.pyqtSlot(list)
    def analyzeAll(self, seriesInfo):
        lock = QtCore.QMutexLocker(self.lock)
        self.runningStatus.emit(True)
        self.progressInit.emit(0, 0)
#        self.progress.emit(1)
        if len(seriesInfo) == 0:
            self.progress.emit(len(seriesInfo) + 3)
            self.runningStatus.emit(False)
            return
        settings = QtCore.QSettings(QtCore.QSettings.UserScope,
                                    QtCore.QString('CTQA'),
                                    QtCore.QString('CTQA-cp'))
        phantom_val = str(settings.value(QtCore.QString('phantom')).toString())
        phantom = PhantomPosition[phantom_val]
        methods = analysisMethods(phantom)
        # gatheringData
        seriesData = []

        for dat in seriesInfo:

            dat['phantomDefinition'] = phantom
            dat['analysisMethods'] = methods
            if 'offset' in dat:
                seriesData.append(dat)
        if len(seriesData) == 0:
            self.progress.emit(len(seriesInfo) + 3)
            self.runningStatus.emit(False)
            return
        nAnalysis = len(seriesData) * 3

        # constructing setup
        if MULTI_PRO:
            ncpu = cpu_count()
            if ncpu < 1:
                ncpu = 1
            self.processes = []
            for _ in range(ncpu):
                self.processes.append(Process(target=analyse_MP,
                                              args=(self.setup_queue,
                                                    self.data_queue,
                                                    self.result_queue,
                                                    self.stop_queue)))

            for p in self.processes:
                p.daemon = True
                p.start()
            for dat in seriesData:
                self.setup_queue.put(dat)
            for _ in self.processes:
                self.stop_queue.put('STOP')
            self.progressInit.emit(0, nAnalysis)
            teller = 0
            for res in iter_queue(self.result_queue,
                                  stop_count=len(self.processes)):
                suid, acn = res.ID
                if not res.success:
                    try:
                        self.logMessage.emit(res.message)
                    except AttributeError:
                        pass
                self.analysis.emit(suid, acn, res)

                teller += 1
                self.progress.emit(teller)

            for p in self.processes:
                p.join(.5)
            for p in self.processes:
                if p.is_alive():
                    p.terminate()
            self.processes = []
            self.progressInit.emit(0, 1)
            self.progress.emit(2)
            self.runningStatus.emit(False)

        else:
            self.progressInit.emit(0, nAnalysis)
            for ind, dat in enumerate(seriesData):
                for func, args in analysisSetUp(dat):
                    res = apply(func, args)
                    suid, acn = res.ID
                    if not res.success:
                        self.logMessage.emit(res.message)
                    self.analysis.emit(suid, acn, res)
                    self.progress.emit(ind + 1)

            self.progressInit.emit(0, 1)
            self.progress.emit(2)
            self.runningStatus.emit(False)

        self.runningStatus.emit(False)


    @QtCore.pyqtSlot(Series)
    def analyze(self, series):
        lock = QtCore.QMutexLocker(self.lock)
        if series.analysis.valid:
            return
        self.runningStatus.emit(True)
        self.progressInit.emit(0, 3)
        self.progress.emit(1)

        anas = analysisSetUp(series)
        self.progress.emit(2)
        jobs = []
        for ana in anas:
            jobs.append(apply(ana[0], ana[1]))
        self.progress.emit(3)
        suid, acn = series.ID, series.aqusitionNumber
        self.analysis.emit(suid, acn, jobs)
        self.runningStatus.emit(False)
        self.progress.emit(4)

    def __del__(self):
        for p in self.processes:
            if p.is_alive():
                p.terminate()
