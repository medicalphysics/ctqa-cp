
"""
Created on Fri Nov 15 15:34:26 2013

@author: erlean
"""

from PyQt4 import QtCore
from pyMime import PyMimeData


class Model(QtCore.QAbstractItemModel):
    def __init__(self, manager, parent=None):
        super(Model, self).__init__(parent)
        self.manager = manager
        self.setSupportedDragActions(QtCore.Qt.CopyAction)
        self.manager.modelChanged.connect(self.layoutChanged)
        self.manager.modelAboutToChange.connect(self.layoutAboutToBeChanged)
        self.manager.dataChanged.connect(self.itemDataChanged)

    def getItem(self, index):
        if index.isValid():
            item = index.internalPointer()
        else:
            item = self.manager.rootItem
        return item

    def indexFromItem(self, row, column, parentItem):
        item = parentItem.child(row)
        return self.createIndex(row, column, item)

    def index(self, row, column, parentIndex):
        if parentIndex.isValid() and parentIndex.column() != 0:
            return QtCore.QModelIndex()

        parentItem = self.getItem(parentIndex)
        item = parentItem.child(row)
        return self.createIndex(row, column, item)

    def parent(self, index):
        if not index.isValid():
            return QtCore.QModelIndex()
        item = self.getItem(index)
        parentItem = item.parent
        if parentItem is self.manager.rootItem:
            return QtCore.QModelIndex()
        row = parentItem.childNumber()
        return self.createIndex(row, 0, parentItem)

    def rowCount(self, parentIndex):
        parent = self.getItem(parentIndex)
        return parent.childCount()

    def columnCount(self, parentIndex):
        return self.getItem(parentIndex).dataCount()

    # for emitting datachanged for external update of data
    @QtCore.pyqtSlot(int, int, object)
    def itemDataChanged(self, row, column, item):
        index = self.createIndex(row, column, item)
        self.dataChanged.emit(index, index)

    def data(self, index, role):
        item = index.internalPointer()
        column = index.column()
        return QtCore.QVariant(item.data(column, role))

#    def hasChildren(self, index):
#        if index.isValid():
#            return index.internalPointer().childrenCount() != 0
#        else:
#            return False

    def setData(self, index, value, role):
        column = index.column()
        item = self.getItem(index)
        success = self.manager.setData(item, column, value.toPyObject(), role)
        if success:
            self.dataChanged.emit(index, index)
        return success

    def headerData(self, section, orientation, role):
        return QtCore.QVariant(self.manager.headerData(section,
                                                       orientation, role))

    def setHeaderData(self, section, orientation, value, role):
        return False

    def mimeTypes(self):
        return [u'application/uri-list', u'application/x-dicomitemdata']

    def mimeData(self, indices):
        qurls = []
        items = []
        info = ""
        for ind in indices:
            if ind.column() == 0:  # Only need one column per item
                item = self.getItem(ind)
                if item is not self.manager.rootItem:
                    items.append(ind)
                    paths = self.manager.getImagePaths(item)
                    info += self.manager.getInfoTableHtml(item)
                    for path in paths:
                        qurls.append(QtCore.QUrl.fromLocalFile(path))
        if len(items) == 0:
            items = None
        mime = PyMimeData(items)
        if len(qurls) > 0:
            mime.setUrls(qurls)
        if info != "":
            mime.setHtml(info)
        return mime

    def flags(self, index):
        if index.isValid():
            item = self.getItem(index)
            return item.flags(index.column())
        return 0

    def fetchMore(self, parentIndex):
        pass

    def canFetchMore(self, parentIndex):
        return False

    def sort(self, column, order):
        self.layoutAboutToBeChanged.emit()
#        self.beginLayoutChanged()
        fIndex = self.persistentIndexList()
        fItem = [item.internalPointer() for item in fIndex]
        self.manager.sort(column, order)
        tIndex = []
        for item in fItem:
            row = item.childNumber()
            tIndex.append(self.createIndex(row, 0, item))
        self.changePersistentIndexList(fIndex, tIndex)
#        self.endLayoutChanged()
        self.layoutChanged.emit()
