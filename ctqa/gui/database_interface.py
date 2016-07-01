# -*- coding: utf-8 -*-

import os
from PyQt4 import QtCore, QtGui
from ctqa.database import Database


class DatabaseInterface(object):
    def __init__(self, path=None):
        self.path = None
        self._db = None
        if path is None:
            path = os.path.join(os.path.abspath(QtCore.QDir().homePath()), 'ctqa_database.h5')

            os.path.abspath(path)

#        self.trigger = QtCore.QTimer()
#        self.trigger.timeout.connect(self.set_database_file)
#        self.trigger.setSingleShot(True)
#        self.trigger.start()

    @QtCore.pyqtSlot()
    @QtCore.pyqtSlot(str)
    def set_database_file(path=None):

