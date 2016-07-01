# -*- coding: utf-8 -*-
"""
Created on Fri May 20 12:11:33 2016

@author: erlean
"""
import sys
from PyQt4 import QtGui, QtCore
from ctqa.version import VERSION


class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('CTQA ' + VERSION)
        self.setupUi()

    def setupUi(self):
        self.setCentralWidget(QtGui.QSplitter())
        self





def start():
    app = QtGui.QApplication(sys.argv)
    app.setOrganizationName("SSHF")

    app.setApplicationName("CTQA")
    win = MainWindow()
    win.show()
    return app.exec_()






if __name__ == '__main__':
    start()
