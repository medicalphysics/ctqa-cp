"""
Created on Tue Dec 10 10:24:04 2013

@author: erlean
"""

from PyQt4 import QtCore, QtGui
import platform
from displayWidgets import BaseWidget, DicomView
from dataManager import Series
from qtutils import qUrlToStr, qStringToStr
import resources

try:
    IS_WIN8 = platform.platform()[:9] == 'Windows-8'
except:
    IS_WIN8 = False


Rect = QtCore.QRect
Pos = QtCore.QPoint
Size = QtCore.QSize


class Display(QtGui.QWidget):
    transistUp = QtCore.pyqtSignal()
    transistDown = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super(Display, self).__init__(parent)

        self.setMinimumSize(50, 50)
        self.setAcceptDrops(True)

        # reultsWidgets
        self.wid1 = ResultWidget(self)
        self.wid2 = ResultWidget(self)
        self.wid3 = ResultWidget(self)
        self.wid4 = ResultWidget(self)
        for wid in [self.wid1, self.wid2, self.wid3, self.wid4]:
            wid.installEventFilter(self)

        self.wid1.closeButton.clicked.connect(self.closeTransist)
        self.wid2.closeButton.clicked.connect(self.closeTransist)
        self.wid3.closeButton.clicked.connect(self.closeTransist)
        self.wid4.closeButton.clicked.connect(self.closeTransist)

        self.setupStateMachine()

    @QtCore.pyqtSlot()
    def clearAll(self):
        for wid in [self.wid1, self.wid2, self.wid3, self.wid4]:
            wid.closeAnalysis()
        self.closeTransist()

    def resizeEvent(self, ev):
        self.assignStateProperties()
        size = self.size()
        w = size.width()
        h = size.height()
        if not self.wid4.size().isNull():
            self.wid1.setGeometry(Rect(Pos(0, 0), size / 2))
            self.wid2.setGeometry(Rect(Pos(w / 2, 0), size / 2))
            self.wid3.setGeometry(Rect(Pos(0, h / 2), size / 2))
            self.wid4.setGeometry(Rect(Pos(w / 2, h / 2), size / 2))
        elif not self.wid2.size().isNull():
            self.wid1.setGeometry(Rect(Pos(0, 0), Size(w / 2, h)))
            self.wid2.setGeometry(Rect(Pos(w / 2, 0), Size(w / 2, h)))
            self.wid3.setGeometry(Rect(Pos(0, h), Size(0, 0)))
            self.wid4.setGeometry(Rect(Pos(w, h), Size(0, 0)))
        else:
            self.wid1.setGeometry(Rect(Pos(0, 0), size))
            self.wid2.setGeometry(Rect(Pos(w, 0), Size(0, 0)))
            self.wid3.setGeometry(Rect(Pos(0, h), Size(0, 0)))
            self.wid4.setGeometry(Rect(Pos(w, h), Size(0, 0)))

        super(Display, self).resizeEvent(ev)

    def showEvent(self, ev):
        self.assignStateProperties()
        super(Display, self).showEvent(ev)

    def assignStateProperties(self):
        size = self.size()
        w = size.width()
        h = size.height()

        # state1
        self.state1.assignProperty(self.wid1, 'geometry', Rect(Pos(0, 0),
                                                               size))
        self.state1.assignProperty(self.wid2, 'geometry', Rect(Pos(w, 0),
                                                               Size(0, 0)))
        self.state1.assignProperty(self.wid3, 'geometry', Rect(Pos(0, h),
                                                               Size(0, 0)))
        self.state1.assignProperty(self.wid4, 'geometry', Rect(Pos(w, h),
                                                               Size(0, 0)))

        self.state2.assignProperty(self.wid1, 'geometry', Rect(Pos(0, 0),
                                                               Size(w / 2, h)))
        self.state2.assignProperty(self.wid2, 'geometry', Rect(Pos(w / 2, 0),
                                                               Size(w / 2, h)))
        self.state2.assignProperty(self.wid3, 'geometry', Rect(Pos(0, h),
                                                               Size(0, 0)))
        self.state2.assignProperty(self.wid4, 'geometry', Rect(Pos(w, h),
                                                               Size(0, 0)))

        self.state3.assignProperty(self.wid1, 'geometry', Rect(Pos(0, 0),
                                                               size / 2))
        self.state3.assignProperty(self.wid2, 'geometry', Rect(Pos(w / 2, 0),
                                                               size / 2))
        self.state3.assignProperty(self.wid3, 'geometry', Rect(Pos(0, h / 2),
                                                               size / 2))
        self.state3.assignProperty(self.wid4, 'geometry',
                                   Rect(Pos(w / 2, h / 2), size / 2))

    def setupStateMachine(self):
        # state Machine
        self.stateMachine = QtCore.QStateMachine()
        self.state1 = QtCore.QState(self.stateMachine)
        self.state2 = QtCore.QState(self.stateMachine)
        self.state3 = QtCore.QState(self.stateMachine)
        self.assignStateProperties()

        # animation
        wids = [self.wid1, self.wid2, self.wid3, self.wid4]
        easingCurve = QtCore.QEasingCurve(QtCore.QEasingCurve.OutBack)
        t12 = self.state1.addTransition(self.transistUp, self.state2)
        for wid in wids:
            a12 = QtCore.QPropertyAnimation(wid, 'geometry', self.state1)
            a12.setEasingCurve(easingCurve)
            if not IS_WIN8:
                t12.addAnimation(a12)
        t23 = self.state2.addTransition(self.transistUp, self.state3)
        for wid in wids:
            a23 = QtCore.QPropertyAnimation(wid, 'geometry', self.state2)
            a23.setEasingCurve(easingCurve)
            if not IS_WIN8:
                t23.addAnimation(a23)
        t32 = self.state3.addTransition(self.transistDown, self.state2)
        for wid in wids:
            a32 = QtCore.QPropertyAnimation(wid, 'geometry', self.state3)
            a32.setEasingCurve(easingCurve)
            if not IS_WIN8:
                t32.addAnimation(a32)
        t21 = self.state2.addTransition(self.transistDown, self.state1)
        for wid in wids:
            a21 = QtCore.QPropertyAnimation(wid, 'geometry', self.state2)
            a21.setEasingCurve(easingCurve)
            if not IS_WIN8:
                t21.addAnimation(a21)
        self.stateMachine.setInitialState(self.state1)
        self.stateMachine.start()

    def openTransist(self):
        wids = [self.wid1, self.wid2, self.wid3, self.wid4]
        vis_wids = [wid for wid in wids if not wid.size().isNull()]
        data_wids = [wid.hasData for wid in vis_wids]
        if all(data_wids):
            self.transistUp.emit()

    @QtCore.pyqtSlot()
    def closeTransist(self):
        if not self.wid4.size().isNull():
            if not self.wid3.hasData and not self.wid4.hasData:
                self.transistDown.emit()

        if not self.wid2.size().isNull():
            if not any([self.wid2.hasData, self.wid3.hasData,
                        self.wid4.hasData]):
                self.transistDown.emit()
        return

    @QtCore.pyqtSlot(str, int, list, list)
    def updateAnalysis(self, suid, an, paths, analysis):
        wids = [self.wid1, self.wid2, self.wid3, self.wid4]
        for wid in wids:
            if wid.hasData:
                if wid.sID == suid and wid.aID == an:
                    wid.setData(analysis, suid, an, True)
                    wid.setImages(paths, suid, an)

    def dragEnterEvent(self, ev):
        if ev.source() != self:
            mime = ev.mimeData()
            try:
                mime.instance()
            except AttributeError:
                if mime.hasUrls():
                    ev.accept()
                    self.assignStateProperties()
                    self.openTransist()
            else:
                ev.accept()
                self.assignStateProperties()
                self.openTransist()

    def dragLeaveEvent(self, ev):
        ev.accept()
        self.closeTransist()

#    def dragMoveEvent(self, ev):
#        print self.childAt(ev.pos())

    def eventFilter(self, obj, ev):
        if ev.type() == QtCore.QEvent.DragEnter:
            return True
        elif ev.type() == QtCore.QEvent.DragMove:
            return True
        elif ev.type() == QtCore.QEvent.DragLeave:
            return False
        return False

    def dropEvent(self, ev):
        mime = ev.mimeData()
        hasSeries = False
        try:
            items = mime.instance()
        except AttributeError:
            pass
        else:
            for itemIndex in items:
                item = itemIndex.internalPointer()
                if isinstance(item, Series):
                    hasSeries = True
                    break
        pos = ev.pos()
        wids = [self.wid1, self.wid2, self.wid3, self.wid4]
        if hasSeries:
            for wid in wids:
                if wid.geometry().contains(pos):
                    suid, acid = item.ID, item.aqusitionNumber
                    wid.setData(item.analysis.analysis, suid, acid)
                    paths = item.getImagePaths()
                    wid.setImages(paths, suid, acid)
                else:
                    wid.setActive(False)
            ev.accept()
        elif mime.hasUrls():

            paths = [url.toLocalFile() for url in mime.urls()]
            for wid in wids:
                if wid.geometry().contains(pos):
                    wid.setImages(paths, None, -1, True)
                else:
                    wid.setActive(False)
            ev.accept()
        else:
            ev.ignore()
        self.closeTransist()

    def mouseReleaseEvent(self, ev):
        pos = ev.pos()
        wids = [self.wid1, self.wid2, self.wid3, self.wid4]
        for wid in wids:
            hasFocus = wid.geometry().contains(pos)
            wid.setActive(hasFocus)
        super(Display, self).mouseReleaseEvent(ev)


class TabBar(QtGui.QTabBar):
    def __init__(self, parent=None):
        super(TabBar, self).__init__(parent)
        self.setObjectName(QtCore.QString('resulttab'))
        self.setDrawBase(False)

    def mouseReleaseEvent(self, e):
        super(TabBar, self).mouseReleaseEvent(e)
        e.setAccepted(False)


class ResultWidget(QtGui.QTabWidget):
    # signal to higlight current series in tree
    seriesHighLight = QtCore.pyqtSignal(str, int)

    def __init__(self, parent=None):
        super(ResultWidget, self).__init__(parent)
        self.setObjectName(QtCore.QString('result'))

        self.setAcceptDrops(False)
        self.hasData = False
        self.setTabBar(TabBar(self))
        msg = "Drop a serie here to display analysis results \n"
        msg += "or drop images to only view images."
        self.addTab(QtGui.QLabel(msg), "")
        self.sID = None
        self.aID = -1
        self.styleHigh = """ResultWidget::pane {border-width: 3px;
                            border-style: solid;
                            }
                            ResultWidget::tab-bar {left: 30px; top: 3px;
                            right: 3px;
                            }
                            QTabBar::tab {border: 1px solid palette(mid);
                            border-top-left-radius: 4px;
                            border-top-right-radius: 4px;
                            min-width: 8ex;
                            padding: 5px;
                            }
                            QTabBar::tab:selected {
                            background: palette(light);
                            }
                            QTabBar::tab:!selected {margin-top: 5px;
                            background: palette(window);
                            }
                            ResultWidget {border-width: 3px;
                            border-style: solid;
                            border-top-color: qlineargradient(x1: 0, y1: 1,
                            x2: 0, y2: 0, stop:0 rgba(126, 0, 255, 56),
                            stop:1 rgba(255, 255, 255, 0));
                            border-right-color: qlineargradient(x1: 0, y1: 0,
                            x2: 1, y2: 0, stop:0 rgba(126, 0, 255, 56),
                            stop:1 rgba(255, 255, 255, 0));
                            border-left-color: qlineargradient(x1: 1, y1: 0,
                            x2: 0, y2: 0, stop:0 rgba(126, 0, 255, 56),
                            stop:1 rgba(255, 255, 255, 0));
                            border-bottom-color: qlineargradient(x1: 0, y1: 0,
                            x2: 0, y2: 1, stop:0 rgba(126, 0, 255, 56),
                            stop:1 rgba(255, 255, 255, 0));
                            }
                            Legend QWidget {
                            background-color: rgba(255,255,255,0);
                            }"""
        self.styleLow = """ResultWidget::pane {border-width: 3px;
                           border-style: solid;
                           }
                           ResultWidget::tab-bar {left: 30px; top: 3px;
                           right: 3px;
                           }
                           QTabBar::tab {border: 1px solid palette(mid);
                           border-top-left-radius: 4px;
                           border-top-right-radius: 4px;
                           min-width: 8ex;
                           padding: 5px;
                           }
                           QTabBar::tab:selected {
                           background: palette(window);
                           }
                           QTabBar::tab:!selected {margin-top: 5px;
                           background: palette(window);
                           }
                           ResultWidget {border-width: 3px;
                           border-style: solid;
                           }
                           Legend QWidget {
                           background-color: rgba(255,255,255,0);
                           }"""
        self.setStyleSheet(self.styleLow)
        self.tabBar().hide()
        # Close button
        icon = QtGui.QIcon(':Icons/Icons/close.png')
        name = 'Close Analysis'
        self.closeButton = QtGui.QPushButton(self)
        self.closeButton.setGeometry(6, 9, 16, 16)
        self.closeButton.setFlat(True)
        self.closeButton.setIcon(icon)
        self.closeButton.setIconSize(QtCore.QSize(16, 16))
        self.closeButton.setToolTip(name)
        self.closeButton.clicked.connect(self.closeAnalysis)
        self.closeButton.setVisible(False)

    @QtCore.pyqtSlot()
    def closeAnalysis(self):
        wids = [self.widget(i) for i in range(self.count())]
        self.clear()
        while len(wids) > 0:
            wids[0].setParent(None)
            wids[0].deleteLater()
            del wids[0]
        self.hasData = False
        self.setActive(self.hasData)

    def setActive(self, active):
        if active and self.hasData:
            self.setStyleSheet(self.styleHigh)
            self.seriesHighLight.emit(self.sID, self.aID)
        else:
            self.setStyleSheet(self.styleLow)
        self.tabBar().setVisible(self.hasData)
        self.closeButton.setVisible(self.hasData)

    def setData(self, analysisList, sID, aID, clear=True):
        if clear:
            self.closeAnalysis()
        self.sID = str(sID)
        self.aID = int(aID)
        for ana in analysisList:
            self.addTab(BaseWidget(ana), ana.name)
        data = len(analysisList) > 0
        self.hasData = data
        self.setActive(data)

    def setImages(self, paths, sID, aID, clear=False):
        if clear:
            self.closeAnalysis()
        self.sID = str(sID)
        self.aID = int(aID)
        imWid = DicomView(paths, parent=self)
        self.addTab(imWid, "Images")
        self.hasData = True
        self.setActive(self.hasData)

    def paintEvent(self, e):
        opt = QtGui.QStyleOption()
        opt.initFrom(self)
        p = QtGui.QPainter(self)
        self.style().drawPrimitive(QtGui.QStyle.PE_Widget, opt, p, self)

    def mouseReleaseEvent(self, e):
        e.setAccepted(False)
