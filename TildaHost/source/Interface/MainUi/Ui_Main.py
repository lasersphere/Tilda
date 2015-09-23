# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_Main.ui'
#
# Created: Tue Sep 22 19:11:26 2015
#      by: PyQt5 UI code generator 5.3.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_TildaMainWindow(object):
    def setupUi(self, TildaMainWindow):
        TildaMainWindow.setObjectName("TildaMainWindow")
        TildaMainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(TildaMainWindow)
        self.centralwidget.setObjectName("centralwidget")
        TildaMainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(TildaMainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        self.menuTilda_MainWindow = QtWidgets.QMenu(self.menubar)
        self.menuTilda_MainWindow.setObjectName("menuTilda_MainWindow")
        self.menuView = QtWidgets.QMenu(self.menubar)
        self.menuView.setObjectName("menuView")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        TildaMainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(TildaMainWindow)
        self.statusbar.setObjectName("statusbar")
        TildaMainWindow.setStatusBar(self.statusbar)
        self.actionTracks = QtWidgets.QAction(TildaMainWindow)
        self.actionTracks.setObjectName("actionTracks")
        self.actionScan_Control = QtWidgets.QAction(TildaMainWindow)
        self.actionScan_Control.setObjectName("actionScan_Control")
        self.actionVersion = QtWidgets.QAction(TildaMainWindow)
        self.actionVersion.setObjectName("actionVersion")
        self.menuView.addAction(self.actionTracks)
        self.menuView.addAction(self.actionScan_Control)
        self.menuHelp.addAction(self.actionVersion)
        self.menubar.addAction(self.menuTilda_MainWindow.menuAction())
        self.menubar.addAction(self.menuView.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(TildaMainWindow)
        QtCore.QMetaObject.connectSlotsByName(TildaMainWindow)

    def retranslateUi(self, TildaMainWindow):
        _translate = QtCore.QCoreApplication.translate
        TildaMainWindow.setWindowTitle(_translate("TildaMainWindow", "Tilda"))
        self.menuTilda_MainWindow.setTitle(_translate("TildaMainWindow", "File"))
        self.menuView.setTitle(_translate("TildaMainWindow", "Tools"))
        self.menuHelp.setTitle(_translate("TildaMainWindow", "Help"))
        self.actionTracks.setText(_translate("TildaMainWindow", "Tracks"))
        self.actionScan_Control.setText(_translate("TildaMainWindow", "Scan Control"))
        self.actionVersion.setText(_translate("TildaMainWindow", "Version"))

