
"""
Created on Fri Nov 15 15:27:36 2013

@author: erlean
"""

from PyQt4 import QtCore, QtGui
from dataManager import Series


#only allow selection of siblings
class SelectionModel(QtGui.QItemSelectionModel):
    def __init__(self, model, parent=None):
        super(SelectionModel, self).__init__(model, parent)

    @QtCore.pyqtSlot(QtCore.QModelIndex,
                     QtGui.QItemSelectionModel.SelectionFlag)
    @QtCore.pyqtSlot(QtGui.QItemSelection,
                     QtGui.QItemSelectionModel.SelectionFlag)
    def select(self, selection, flag):
        if isinstance(selection, QtCore.QModelIndex):
            super(SelectionModel, self).select(selection, flag)
            return

        indexes = selection.indexes()

        if len(indexes) < 2:
            super(SelectionModel, self).select(selection, flag)
            return

        parent = indexes[0].parent()
        valid_indexes = []
        for index in indexes:
            if index.parent() == parent:
                valid_indexes.append(index)
        selection = QtGui.QItemSelection(valid_indexes[0], valid_indexes[-1])
        super(SelectionModel, self).select(selection, flag)


class TreeView(QtGui.QTreeView):
    dataImportRequest = QtCore.pyqtSignal(list)
    logMessage = QtCore.pyqtSignal(QtCore.QString)

    def __init__(self, parent=None):
        super(TreeView, self).__init__(parent)
        self.setUniformRowHeights(False)
        self.setIconSize(QtCore.QSize(16, 16))
        self.setAnimated(True)
        self.setDragDropMode(self.DragDrop)
        self.setDragEnabled(True)
        self.setSortingEnabled(True)
        self.setHeaderHidden(False)
        self.setSelectionMode(self.ExtendedSelection)
        self.hasData = False

    def dragMoveEvent(self, ev):
        ev.accept()

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls() and ev.source() is not self:
            ev.setDropAction(QtCore.Qt.CopyAction)
            ev.accept()
        else:
            ev.ignore()

    def setModel(self, model):
        super(TreeView, self).setModel(model)
        self.setSelectionModel(SelectionModel(model, self))

    def dropEvent(self, ev):
        if ev.mimeData().hasUrls():
            paths = ev.mimeData().urls()
            self.dataImportRequest.emit(paths)
            ev.accept()
            self.hasData = True
        else:
            ev.ignore()

    def paintEvent(self, ev):
        super(TreeView, self).paintEvent(ev)
        if not self.hasData:
            p = QtGui.QPainter(self.viewport())
            r = self.rect()
            p.drawText(r, QtCore.Qt.AlignCenter,
                       "Drop Folders or DiCOM files here...")

    @QtCore.pyqtSlot(Series)
    def highLightSeries(self, series):
        model = self.model()
        row = series.childNumber()
        index = model.createIndex(row, 0, series)
        self.selectionModel().select(index,
                                     QtGui.QItemSelectionModel.ClearAndSelect)
