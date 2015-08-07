"""
Created on Fri Nov 29 13:49:26 2013

@author: erlean
"""
from PyQt4 import QtCore, QtGui
import imageTools
from dataImporter import validTags, extraTags
import dicom


infoTags = {}
## excluding some tags from the information tab.
infoTags_ex = ['imageID', 'imageLocation', 'mas',
               'ctdiVol', 'studyID', 'exposure', 'exposuretime']
for key, val in dict(validTags.items() + extraTags.items()).items():
    if not key in infoTags_ex:
        infoTags[key] = val


class BaseItem(object):
    def __init__(self, parent=None, ID=None):
        self.parent = parent
        self.ID = ID
        self.children = []
        self.displayData = []
        if parent is not None:
            self.parent.addChild(self)

    def ID(self):
        return self.ID

    def addChild(self, child):
        self.children.append(child)

    def childNumber(self):
        if self.parent is not None:
            return self.parent.children.index(self)
        return 0

    def removeChild(self, child):
        index = self.children.index(child)
        del self.children[index]

    def child(self, row):
        return self.children[row]

    def childCount(self):
        return len(self.children)

    def dataCount(self):
        lenght = len(self.displayData)
        if lenght > 0:
            return lenght
        return 1

    def flags(self, column=0):
        qt = QtCore.Qt
        return qt.ItemIsSelectable | qt.ItemIsEnabled

    def setData(self, column, value, role):
        if role == QtCore.Qt.DisplayRole:
            while len(self.displayData) <= column:
                    self.displayData.append("")
            self.displayData[column] = value
            return True
        return False

    def data(self, column, role):
        if role == QtCore.Qt.DisplayRole:
            if column < len(self.displayData):
                return QtCore.QVariant(self.displayData[column])
        return QtCore.QVariant()


class Root(BaseItem):
    def __init__(self):
        super(Root, self).__init__()

#    def flags(self):
#        return QtCore.Qt.NoItemFlags


class Series(BaseItem):
    def __init__(self, parent, ID, imagePath, data, headerTags):
        super(Series, self).__init__(parent, ID)
#        try:
#            print data[validTags['aqusitionNumber']]
#            self.aqusitionNumber = int(data[validTags['aqusitionNumber']])
#        except ValueError:
#            self.aqusitionNumber = 0
        self.aqusitionNumber = data[validTags['aqusitionNumber']]

        InfoContainer(self, None, data)
        AP = data[validTags['patientPosition']][:2] == 'HF'
        self.imageLocation = Localization(self, data[validTags['coordID']], AP)
        self.analysis = Analysis(self)
        self.imageContainer = ImageContainer(self, None)

        self.coordinateID = str(data[validTags['coordID']])
        self.patientOrientation = str(str(data[validTags['patientPosition']]))
        for tag in headerTags:
            try:
                if tag == extraTags['seriesDate']:
                    date = data[tag]
                    qdate = QtCore.QDate(int(date[:4]), int(date[4:6]),
                                         int(date[6:]))
                    self.displayData.append(qdate.toString("ddd dd.MM.yyyy"))
                elif tag == extraTags['seriesTime']:
                    time = str(data[tag])
                    qtime = QtCore.QTime(int(time[:2]), int(time[2:4]),
                                         int(time[4:6]))
                    self.displayData.append(qtime.toString("hh:mm:ss"))
                elif tag == extraTags['seriesDescription']:
                    if tag not in data:
                        self.displayData.append(QtCore.QString("Series"))
                    elif len(data[tag].strip()) > 0:
                        self.displayData.append(QtCore.QString(str(data[tag])))
                    else:
                        self.displayData.append(QtCore.QString("Series"))
                else:
                    self.displayData.append(QtCore.QString(str(data[tag])))
            except KeyError:
                self.displayData.append(QtCore.QString(""))
        self.addImage(imagePath, data, headerTags)

    def addImage(self, imagePath, data, headerTags):
#        print data
#        import sys
#        sys.exit(0)
        IDs = [str(child.ID) for child in self.imageContainer.children]
        ID = str(data[validTags['imageID']])
        if ID in IDs:
            return

        if self.aqusitionNumber != data[validTags['aqusitionNumber']]:
            parent = self.parent
            Series(parent, ID, imagePath, data, headerTags)
        else:
            Image(self.imageContainer, ID, imagePath, data)

    def setAnalysis(self, analysis):
        self.analysis.setAnalysis(analysis)

    def flags(self, column=0):
        return super(Series,
                     self).flags() | QtCore.Qt.ItemIsDragEnabled

    def getImagePaths(self):
        return [child.path for child in self.imageContainer.children]

    def getImagePos(self):
        return [child.pos for child in self.imageContainer.children]

    def getAnalysisData(self):
        dat = {}
        dat['AP'] = self.imageLocation.AP
        dat['paths'] = [p for p in self.getImagePaths()]
        dat['zloc'] = self.getImagePos()
        try:
            dat['offset'] = float(self.imageLocation.displayData[1])
        except ValueError:
            pass
        dat['ID'] = (str(self.ID), self.aqusitionNumber)
        return dat


class InfoItem(BaseItem):
    def __init__(self, parent, ID):
        super(InfoItem, self).__init__(parent, ID)

    def getTableDataTxt(self):
        txt = QtCore.QString("")
        n = len(self.displayData)
        for ind, d in enumerate(self.displayData):
            txt.append(d)
            if ind < n - 1:
                txt.append("\t")
        return txt

    def flags(self, column=0):
        return super(InfoItem,
                     self).flags() | QtCore.Qt.ItemIsDragEnabled


class InfoContainer(BaseItem):
    def __init__(self, parent, ID, info):
        super(InfoContainer, self).__init__(parent, ID)
        self.displayData = ['Series Info', '']
        for val in infoTags.values():
            try:
                childDisplayData = [dicom.datadict.dictionary_description(val),
                                    QtCore.QString(str(info[val]))]
            except KeyError:
                pass
            else:
                child = InfoItem(self, None)
                child.displayData = childDisplayData
        self.children.sort(key=lambda x: x.displayData[0])

    def getTableDataHtml(self):
        txt = QtCore.QString("")
        n = len(self.children)
        for ind, child in enumerate(self.children):
            txt.append(child.getTableDataTxt())
            if ind < n-1:
                txt.append("\n")
        html = QtCore.Qt.escape(txt)
        html.replace("\t", "<td>")
        html.replace("\n", "\n<tr><td>")
        html.prepend("<table>\n<tr><td>")
        html.append("\n</table>")
        return html

    def flags(self, column=0):
        return super(InfoContainer,
                     self).flags() | QtCore.Qt.ItemIsDragEnabled


class Localization(BaseItem):
    def __init__(self, parent, ID, AP):
        super(Localization, self).__init__(parent, ID)
        self.displayData = [QtCore.QString('Phantom Position'),
                            0.,
                            QtCore.QString('Not analysed yet...'),
                            QtCore.QString('')]
        self.automaticCorrectionMode = True
        self.automaticCorrection = None
        self.AP = AP        # AnteriorPosterior

    def setAutomaticCorrection(self, value, AP):
        if value != value:
            self.displayData[2] = QtCore.QString("Not Found")
            self.automaticCorrection = None
            self.displayData[1] = QtCore.QString("Enter manual value")
            self.displayData[3] = QtCore.QString(str(self.AP))
        else:
            self.automaticCorrection = value
            self.displayData[2] = QtCore.QString("Automatic")
            self.displayData[1] = QtCore.QString(format(
                self.automaticCorrection, '.3f'))
            self.displayData[3] = QtCore.QString(str(AP))
            self.AP = AP

    def setData(self, column, value, role):
        if role == QtCore.Qt.EditRole and column == 1:
            if value == "":
                if self.automaticCorrection is not None:
                    self.displayData[column + 1] = QtCore.QString("Automatic")
                    if abs(float(self.displayData[column]) -
                           self.automaticCorrection) > .001:
                        self.displayData[column] = self.automaticCorrection
                        return True
                else:
                    self.displayData[column + 1] = QtCore.QString("Not Found")
                    self.displayData[column] = QtCore.QString("Enter manual "
                                                              "value")
                return False
            try:
                value = float(value)
            except ValueError:
                return False
            else:
                try:
                    cval = float(self.displayData[column])
                except ValueError:
                    self.displayData[column] = value
                    self.displayData[column + 1] = QtCore.QString("Manual"
                                                                  " entry")
                    return True
                else:
                    if abs(cval - value) > .001:
                        self.displayData[column] = value
                        self.displayData[column + 1] = QtCore.QString("Manual"
                                                                      " entry")
                        return True
                    else:
                        return False
        return super(Localization, self).setData(column, value, role)

    def data(self, column, role):
        if role == QtCore.Qt.ForegroundRole and column == 1:
            if self.displayData[1] == QtCore.QString("Enter manual value"):
                return QtCore.QVariant(QtGui.QBrush(QtGui.QColor(0,
                                                                 0, 0, 127)))
        return super(Localization, self).data(column, role)

    def flags(self, column=0):
        base_flags = super(Localization, self).flags()
        if column == 1:
            return base_flags | QtCore.Qt.ItemIsEditable
        return base_flags


class ImageContainer(BaseItem):
    def __init__(self, parent, ID):
        super(ImageContainer, self).__init__(parent, ID)
        self.displayData = [QtCore.QString(st) for st in ['Images', 'Exposure',
                            'CTDIvol', 'Path', ]]  # must correspond to image

    def flags(self, column=0):
        return super(ImageContainer,
                     self).flags() | QtCore.Qt.ItemIsDragEnabled


class Image(BaseItem):
    def __init__(self, parent, ID, path, data):
        super(Image, self).__init__(parent, ID)
        self.path = path
        self.inCP404 = data['inCP404']
        self.center = data['center']
        self.phantomPos = data['phantomPos']
        self.thumbnail = imageTools.arrayToQImage(data['thumbnail'])
        self.pos = float(data[validTags['imageLocation']][2])
        self.displayData = [self.pos]
        mas = QtCore.QString(str(data.get(extraTags['mas'], 'Not found')))
        ctdi = QtCore.QString(str(data.get(extraTags['ctdiVol'], 'Not found')))
        self.displayData.append(mas)
        self.displayData.append(ctdi)
        self.displayData.append(self.url.toLocalFile())

    @property
    def url(self):
        return QtCore.QUrl.fromLocalFile(self.path)

    def data(self, column, role):
        if role == QtCore.Qt.DecorationRole and column == 0:
            return QtGui.QPixmap().fromImage(self.thumbnail)
        if role == QtCore.Qt.SizeHintRole and column == 0:
            return QtCore.QSize(self.thumbnail.size())
        return super(Image, self).data(column, role)

    def flags(self, column=0):
        return super(Image, self).flags() | QtCore.Qt.ItemIsDragEnabled


class Analysis(BaseItem):
    def __init__(self, parent, ID=None):
        super(Analysis, self).__init__(parent, ID)
        self.analysis = []
        self.valid = False
        self.displayData = [QtCore.QString(st) for st in ['Analysis',
                                                          'Not started']]
        self.uids = []

    def unsetAnalysis(self):
        self.analysis = []
        self.valid = False
        self.uids = []

    def setAnalysis(self, analysis):
        names = [a.name for a in self.analysis]
        if analysis.name in names:
            del self.analysis[names.index(analysis.name)]
            del self.uids[names.index(analysis.name)]
        self.analysis.append(analysis)
        self.analysis.sort(key=lambda x: x.name)
        self.valid = True
        self.uids.append(analysis.imageUids)

    def data(self, column, role):
        if role == QtCore.Qt.DisplayRole and column == 1:
            if self.valid:
                return QtCore.QString('Analysis Finished')
            else:
                return QtCore.QString('Not Analysed')
        else:
            return super(Analysis, self).data(column, role)


class DataManager(QtCore.QObject):
    modelChanged = QtCore.pyqtSignal()
    modelAboutToChange = QtCore.pyqtSignal()
    layoutAboutToBeChanged = QtCore.pyqtSignal()
    layoutChanged = QtCore.pyqtSignal()
    dataChanged = QtCore.pyqtSignal(int, int, object)
    delayedImportFinished = QtCore.pyqtSignal()
    startLocalizer = QtCore.pyqtSignal(Root)
    delayedImportProgressValue = QtCore.pyqtSignal(int)
    delayedImportProgressRange = QtCore.pyqtSignal(int, int)
    exportProgressValue = QtCore.pyqtSignal(int)
    exportProgressRange = QtCore.pyqtSignal(int, int)
    analyze = QtCore.pyqtSignal(Series)
    analyzeAll = QtCore.pyqtSignal(list)
    message = QtCore.pyqtSignal(QtCore.QString, int)
    logMessage = QtCore.pyqtSignal(QtCore.QString)
    seriesHighLight = QtCore.pyqtSignal(Series)
    modelLocked = QtCore.pyqtSignal(bool)
    updateAnalysis = QtCore.pyqtSignal(str, int, list, list)

    def __init__(self, parent=None):
        super(DataManager, self).__init__(parent)
        self.rootItem = Root()
        self.delayedImport = []
        self.modelLock = QtCore.QMutex(mode=QtCore.QMutex.Recursive)

        self.headerTags = [extraTags['seriesDescription'],
                           extraTags['seriesDate'],
                           extraTags['seriesTime'],
                           extraTags['seriesNumber'],
                           validTags['aqusitionNumber'],
                           validTags['patientID']]
        self.rootItem.displayData = []
        for tag in self.headerTags:
            h = dicom.datadict.dictionary_description(tag)
            self.rootItem.displayData.append(QtCore.QString(h))

    def getSeriesData(self):
        return [child.getAnalysisData() for child in self.rootItem.children
                if not child.analysis.valid]

    @QtCore.pyqtSlot()
    def phantomChanged(self):
        self.modelLock.lock()
        for child in self.rootItem.children:
            child.analysis.unsetAnalysis()
        self.analyzeAll.emit(self.getSeriesData())
        self.modelLock.unlock()

    @QtCore.pyqtSlot()
    def importFinished(self):
        self.modelLock.lock()
        n_images = len(self.delayedImport)
        if n_images > 0:
            self.delayedImportProgressRange.emit(0, n_images-1)
            progress = 0
            while len(self.delayedImport) > 0:
                self.addImage(self.delayedImport.pop(0))
                progress += 1
                self.delayedImportProgressValue.emit(progress)
        self.modelLock.unlock()
        self.startLocalizer.emit(self.rootItem)

    @QtCore.pyqtSlot(bool)
    def localizerFinished(self, running):
        if not running:
            self.modelLock.lock()
            self.analyzeAll.emit(self.getSeriesData())
            self.modelLock.unlock()

    @QtCore.pyqtSlot(bool)
    def lockModel(self, lock):
        pass
#        self.modelLock = lock
#        self.modelLocked.emit(self.modelLock)

    def getImagePaths(self, item):
        if isinstance(item, Series):
            return [child.path for child in item.imageContainer.children]
        elif isinstance(item, ImageContainer):
            return [child.path for child in item.children]
        elif isinstance(item, Image):
            return [item.path]
        return []

    def getInfoTableHtml(self, item):
        try:
            tab = item.getTableDataHtml()
        except AttributeError:
            return ""
        else:
            return tab

    @QtCore.pyqtSlot(int, QtCore.Qt.SortOrder)
    def sort(self, column, order):
#        if self.modelLock:
#            self.message.emit("Please wait for analysis to complete", 5000)
#            return
        self.modelLock.lock()
#        self.lockModel(True)
        order == QtCore.Qt.DescendingOrder
        if self.rootItem.dataCount() > column:
            try:
                self.rootItem.children.sort(key=lambda x: int(x.displayData[column]),
                                            reverse=order)
            except ValueError:
                self.rootItem.children.sort(key=lambda x: x.displayData[column],
                                            reverse=order)

        for child in self.rootItem.children:
            if child.imageContainer.dataCount() > column:
                child.imageContainer.children.sort(key=lambda x:
                                                   x.displayData[column],
                                                   reverse=order)
#        self.lockModel(False)
        self.modelLock.unlock()

    def findSeriesByFOR(self, FOR):
        forList = []
        for series in self.rootItem.children:
            if series.coordinateID == FOR:
                forList.append(series)
        return forList

    def findSeries(self, ID, acN):  # finding a series from ID and acusition
        for series in self.rootItem.children:
            if series.ID == ID and series.aqusitionNumber == acN:
                return series
        return None

    def headerData(self, section, orientation, role):
        if orientation == QtCore.Qt.Horizontal:
            if role == QtCore.Qt.DisplayRole:
                return self.rootItem.displayData[section]
        return None

    @QtCore.pyqtSlot(str, int, object)
    def setAnalysis(self, suid, acn, analysis):
        series = self.findSeries(suid, acn)
        series.setAnalysis(analysis)
        row = series.analysis.childNumber()
        column = 0
        self.dataChanged.emit(row, column, series.analysis)
        ana = series.analysis.analysis
        self.updateAnalysis.emit(suid, acn, series.getImagePaths(), ana)

    @QtCore.pyqtSlot(list, float, bool)
    def setAutoCorrection(self, series, value, AP):
        for serie in series:
            imLoc = serie.imageLocation
            imLoc.setAutomaticCorrection(value, AP)
            serie.analysis.unsetAnalysis()
            row = imLoc.childNumber()
            column = 0
            self.dataChanged.emit(row, column, imLoc)

    @QtCore.pyqtSlot(str, int)
    def highLightSeries(self, sID, aID):
        series = self.findSeries(sID, aID)
        if series is None:
            return
        self.seriesHighLight.emit(series)

    @QtCore.pyqtSlot(tuple)
    def addImage(self, info):
#        if self.modelLock:
#            self.delayedImport.append(info)
#            return
#        self.lockModel(True)
        self.modelLock.lock()
        self.modelAboutToChange.emit()
        path, data = info
        ID = str(data[validTags['seriesID']])
        acNumber = data[validTags['aqusitionNumber']]
        existingSeries = self.findSeries(ID, acNumber)
        if existingSeries is None:
            Series(self.rootItem, ID, path, data, self.headerTags)
        else:
            existingSeries.addImage(path, data, self.headerTags)
        self.modelChanged.emit()
#        self.lockModel(False)
        self.modelLock.unlock()

    def setData(self, item, *args):
#        if self.modelLock:
#            self.message.emit("Please wait for analysis to complete", 5000)
#            return False
        self.modelLock.lock()
        success = item.setData(*args)
        #analyse series after data change
        if success and args[2] == QtCore.Qt.EditRole:
            if isinstance(item, Series):
                aList = self.findSeriesByFOR(item.coordinateID)
                for s in aList:
                    s.analysis.unsetAnalysis()
                    s.imageLocation.setData(*args)
                    row = s.imageLocation.childNumber()
                    self.dataChanged.emit(row, args[1],
                                          s.imageLocation)
                self.analyzeAll.emit(self.getSeriesData())
            else:
                while item is not None:
                    item = item.parent
                    if isinstance(item, Series):
                        aList = self.findSeriesByFOR(item.coordinateID)
                        for s in aList:
                            s.analysis.unsetAnalysis()
                            s.imageLocation.setData(*args)
                            row = s.imageLocation.childNumber()
                            self.dataChanged.emit(row, args[1],
                                                  s.imageLocation)
                        self.analyzeAll.emit(self.getSeriesData())
                        break
        self.modelLock.unlock()
        return success

    @QtCore.pyqtSlot(QtCore.QString)
    def exportAnalysisImages(self, dest):
#        if self.modelLock:
#            self.message.emit("Please wait for analysis to complete", 5000)
#            return
        self.exportProgressRange.emit(0, 0)
        flist = []
        for series in self.rootItem.children:
            uids = [u for ana in series.analysis.uids for u in ana]
            for im in series.imageContainer.children:
                if im.ID in uids:
                    flist.append(im.url)
        if len(flist) == 0:
            self.message.emit("No images to export", 5000)
            self.exportProgressValue.emit(1)
            return

        dr = QtCore.QDir(path=dest)
        if not dr.exists():
            self.message.emit("ERROR: Make sure export directory exists", 5000)
            self.exportProgressValue.emit(1)
            return

        filobj = QtCore.QFileInfo()
        qfil = QtCore.QFile('')
        n_files = len(flist)
        n = 1
        self.exportProgressRange.emit(0, n_files)
        for f in flist:
            filobj.setFile(f.toLocalFile())
            fname = filobj.fileName()
            i = 1
            while dr.exists(fname):
                fname = QtCore.QString(filobj.fileName() + '_' + str(i))
                i += 1
            dpath = dr.filePath(fname)
            n += 1
            self.exportProgressValue.emit(n)
            suc = qfil.copy(filobj.filePath(), dpath)
            if suc:
                msg = QtCore.QString('Exported') + filobj.filePath()
                msg += QtCore.QString(' to ') + dpath
            else:
                msg = QtCore.QString('Failed copying ') + filobj.filePath()
                msg += QtCore.QString(': ') + QtCore.QString(qfil.error())
            self.logMessage.emit(msg)
