"""
Created on Tue Apr 23 10:04:59 2013


@author: erlean
"""
from __future__ import unicode_literals

Version = '0.3.4'
USE_THREADS = True  # False for debug

import sys
from PyQt4 import QtCore, QtGui
from multiprocessing import freeze_support
from treeView import TreeView
from dataImporter import DataImporter
from dataManager import DataManager
from analyzer import Analyzer, PhantomPosition
from model import Model
from display import Display
from localizer import Localizer
import resources


class Ctqa(QtGui.QMainWindow):
    exportImagesSignal = QtCore.pyqtSignal(QtCore.QString)

    def __init__(self):
        super(Ctqa, self).__init__()
        self.initUI()
        self.initFlow()

    def initUI(self):
        # window
        self.setWindowTitle('CTQA-cp ' + Version)
        self.setCentralWidget(QtGui.QWidget())
        self.setWindowIcon(QtGui.QIcon(':Icons/Icons/ic.png'))
        self.resize(800, 600)
        # layout
        mainLayout = QtGui.QHBoxLayout(self.centralWidget())
        mainLayout.setSpacing(0)
        mainLayout.setContentsMargins(0, 0, 0, 0)
        splitter = QtGui.QSplitter(QtCore.Qt.Horizontal, self.centralWidget())

        mainLayout.addWidget(splitter)

        # Tree
        self.treeView = TreeView(self)
        splitter.addWidget(self.treeView)
        self.display = Display(self)
        splitter.addWidget(self.display)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 6)

        # statusbar
        self.setStatusBar(QtGui.QStatusBar())
        self.statusBar().setFixedHeight(24)
        self.importProgressBar = ProgressBar('Importing', self)
        self.localizerProgressBar = ProgressBar('Localizing', self)
        self.analyzerProgressBar = ProgressBar('Analysing', self)
        self.exportProgressBar = ProgressBar('Exporting', self)
        self.statusBar().addWidget(self.importProgressBar)
        self.statusBar().addWidget(self.localizerProgressBar)
        self.statusBar().addWidget(self.analyzerProgressBar)
        self.statusBar().addWidget(self.exportProgressBar)

        # Phantom selection
        self.phantomList = PhantomComboBox(self)
        self.statusBar().addPermanentWidget(self.phantomList)

        # exportButton
        analysisImageExport = QtGui.QPushButton()
        analysisImageExport.setFlat(True)
        analysisImageExport.setToolTip(
            "Export images associated with an analysis")
        analysisImageExport.setIcon(
            QtGui.QIcon(':Icons/Icons/exportImages.png'))
        analysisImageExport.clicked.connect(self.exportImages)
        self.statusBar().addPermanentWidget(analysisImageExport)

        # logwidget
        self.logWindow = LogWidget(self)
        logButton = QtGui.QPushButton()
        logButton.setFlat(True)
        logButton.setToolTip("Show log")
        logButton.setIcon(QtGui.QIcon(':Icons/Icons/log.png'))
        logButton.clicked.connect(self.logWindow.toggleVisible)
        self.statusBar().addPermanentWidget(logButton)

    def initFlow(self):
        self.dataThread = QtCore.QThread(self)
        self.modelThread = QtCore.QThread(self)
        self.dataImporter = DataImporter()
        self.analyzer = Analyzer()
        self.dataManager = DataManager()
        self.localizer = Localizer()
        if USE_THREADS:
            self.dataImporter.moveToThread(self.dataThread)
            self.analyzer.moveToThread(self.dataThread)
            self.dataManager.moveToThread(self.modelThread)
            self.localizer.moveToThread(self.modelThread)  # must have same
                                                           # thread as
                                                           # datamanager
        self.model = Model(self.dataManager)

        # setting model
        self.treeView.setModel(self.model)
        # connecting tree to importer
        self.treeView.dataImportRequest.connect(self.dataImporter.importPath)
        self.dataImporter.imageFound.connect(self.dataManager.addImage)
        self.treeView.logMessage.connect(self.logWindow.addMessage)

        # connecting localizer to progress bars
        self.localizer.progress.connect(self.localizerProgressBar.setValue)
        self.localizer.progressInit.connect(self.localizerProgressBar.setRange)

        # connecting analyzer to progress bars and logging widget
        self.analyzer.progress.connect(self.analyzerProgressBar.setValue)
        self.analyzer.progressInit.connect(self.analyzerProgressBar.setRange)
        self.analyzer.logMessage.connect(self.logWindow.addMessage)

        # connecting importer to progress bars and log widget
        self.dataImporter.progress.connect(self.importProgressBar.setValue)
        self.dataImporter.progressInit.connect(self.importProgressBar.setRange)
        self.dataImporter.logMessage.connect(self.logWindow.addMessage)

        # connecting manager to statusbar and log and combobox
        statusBar = self.statusBar()
        self.dataManager.message.connect(statusBar.showMessage)
        self.dataManager.logMessage.connect(self.logWindow.addMessage)
        self.dataManager.modelLocked.connect(self.phantomList.setDisabled)
        self.phantomList.newPhantomSelected.connect(
            self.dataManager.phantomChanged)

        # connecting datamanager and importer to localizer
        self.dataManager.startLocalizer.connect(self.localizer.localize)
        self.dataImporter.importFinished.connect(
            self.dataManager.importFinished)
        self.localizer.runningStatus.connect(self.dataManager.lockModel)
        self.localizer.positionCorrection.connect(
            self.dataManager.setAutoCorrection)
        # connecting datamanager delayed import to progressbar
        self.dataManager.delayedImportProgressRange.connect(
            self.importProgressBar.setRange)
        self.dataManager.delayedImportProgressValue.connect(
            self.importProgressBar.setValue)

        # Connectinmg analyser to manager and localizer and display
        self.localizer.runningStatus.connect(
            self.dataManager.localizerFinished)
        self.analyzer.runningStatus.connect(self.dataManager.lockModel)
        self.analyzer.analysis.connect(self.dataManager.setAnalysis)
        self.analyzer.runningStatus.connect(self.phantomList.setDisabled)
        self.dataManager.analyzeAll.connect(self.analyzer.analyzeAll)
        self.dataManager.analyze.connect(self.analyzer.analyze)
        self.dataManager.updateAnalysis.connect(self.display.updateAnalysis)

        # Connecting displaywidgets to manager and tree
        # For highlighting series
        dWids = [self.display.wid1, self.display.wid2, self.display.wid3,
                 self.display.wid4]
        for dWid in dWids:
            dWid.seriesHighLight.connect(self.dataManager.highLightSeries)
        self.dataManager.seriesHighLight.connect(self.treeView.highLightSeries)

        # connecting analysisImageExport button to manager
        self.exportImagesSignal.connect(self.dataManager.exportAnalysisImages)
        self.dataManager.exportProgressValue.connect(
            self.exportProgressBar.setValue)
        self.dataManager.exportProgressRange.connect(
            self.exportProgressBar.setRange)

        # starting threads
        self.modelThread.start()
        self.dataThread.start()

    @QtCore.pyqtSlot()
    def exportImages(self):
        exmsg = "Select export directory"
        dest = QtGui.QFileDialog().getExistingDirectory(caption=exmsg)
        if len(dest) > 0:
            self.exportImagesSignal.emit(dest)

    def __del__(self):
        self.modelThread.exit()
        self.dataThread.exit()
        self.modelThread.wait()
        self.dataThread.wait()


class PhantomComboBox(QtGui.QComboBox):
    newPhantomSelected = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super(PhantomComboBox, self).__init__(parent)
        self.setEditable(False)
        self.phantoms = [k for k in PhantomPosition.keys()]
        self.phantoms.sort(reverse=True)
        self.currentIndexChanged.connect(self.phantomChanged)
#        settings = QtCore.QSettings(self)
        settings = QtCore.QSettings(QtCore.QSettings.UserScope,
                                    QtCore.QString('CTQA'),
                                    QtCore.QString('CTQA-cp'))
        # finding current settings
        for ph in self.phantoms:
            self.addItem(QtCore.QString(ph))
        if not settings.contains(QtCore.QString('phantom')):
            settings.setValue(QtCore.QString('phantom'), QtCore.QVariant(
                QtCore.QString(self.phantoms[0])))
            self.setCurrentIndex(0)
            settings.sync()

        else:

            current_val = settings.value(QtCore.QString('phantom'))
            current_phantom = str(current_val.toString())
            try:
                ind = self.phantoms.index(current_phantom)
            except ValueError:
                settings.setValue(QtCore.QString('phantom'), QtCore.QVariant(
                    QtCore.QString(self.phantoms[0])))
                settings.sync()
                self.setCurrentIndex(0)
            else:
                self.setCurrentIndex(ind)

    @QtCore.pyqtSlot(int)
    def phantomChanged(self, ind):
        settings = QtCore.QSettings(QtCore.QSettings.UserScope,
                                    QtCore.QString('CTQA'),
                                    QtCore.QString('CTQA-cp'))
        settings.setValue(QtCore.QString('phantom'),
                          QtCore.QVariant(QtCore.QString(self.phantoms[ind])))
        settings.sync()
        self.newPhantomSelected.emit()

    @QtCore.pyqtSlot(bool)
    def setDisabled(self, k):
        self.setEnabled(not k)


class ProgressBar(QtGui.QProgressBar):
    def __init__(self, txt, parent=None):
        super(ProgressBar, self).__init__(parent)
        self.setTextVisible(True)
        f = str("%p% " + txt)
        self.setFormat(f)
        self.hide()
        self.setValue(0)

    @QtCore.pyqtSlot(int, int)
    def setRange(self, start, stop):
        super(ProgressBar, self).setRange(start, stop)
        self.show()
        self.setValue(0)

    @QtCore.pyqtSlot(int)
    def setValue(self, val):
        if val > self.maximum():
            self.hide()
        else:
            super(ProgressBar, self).setValue(val)


class LogWidget(QtGui.QMainWindow):
    closed = QtCore.pyqtSignal()

    def __init__(self, parent):
        super(LogWidget, self).__init__(parent)
        self.resize(320, 240)
        self.log = QtGui.QWidget(self)
        self.horizontalLayout = QtGui.QHBoxLayout(self.log)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setMargin(0)
        self.textEdit = QtGui.QPlainTextEdit(self.log)
        self.textEdit.setReadOnly(True)
        self.textEdit.setMaximumBlockCount(1000)
        self.horizontalLayout.addWidget(self.textEdit)
        self.setCentralWidget(self.log)
        self.setWindowTitle("Log")

    @QtCore.pyqtSlot()
    def toggleVisible(self):
        self.setVisible(not self.isVisible())

    @QtCore.pyqtSlot(QtCore.QString)
    def addMessage(self, txt):
        self.textEdit.appendPlainText(txt)

    def closeEvent(self, ev):
        ev.accept()
        self.hide()
        self.closed.emit()


def main(args):
    app = QtGui.QApplication(args)
    app.setOrganizationName("SSHF")
#    app.setOrganizationDomain("https://code.google.com/p/ctqa-cp/")
    app.setApplicationName("CTQA_cp")
    win = Ctqa()
    win.show()
    return app.exec_()

if __name__ == "__main__":
    freeze_support()
    # exit code 1 triggers a restart
    # Also testing for memory error
    try:
        while main(sys.argv) == 1:
            continue
    except MemoryError, e:
        msg = QtGui.QMessageBox()
        msg.setText("Ouch, CTQA-cp ran out of memory.")
        msg.setIcon(msg.Critical)
        msg.exec_()
    sys.exit(0)
