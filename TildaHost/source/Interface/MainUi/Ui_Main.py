# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_Main.ui'
#
# Created by: PyQt5 UI code generator 5.5
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_TildaMainWindow(object):
    def setupUi(self, TildaMainWindow):
        TildaMainWindow.setObjectName("TildaMainWindow")
        TildaMainWindow.resize(492, 247)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("../../../Tilda256.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        TildaMainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(TildaMainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.label_workdir = QtWidgets.QLabel(self.centralwidget)
        self.label_workdir.setObjectName("label_workdir")
        self.gridLayout.addWidget(self.label_workdir, 0, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 1, 0, 1, 1)
        self.label_workdir_set = QtWidgets.QLabel(self.centralwidget)
        self.label_workdir_set.setObjectName("label_workdir_set")
        self.gridLayout.addWidget(self.label_workdir_set, 0, 1, 1, 1)
        TildaMainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(TildaMainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 492, 21))
        self.menubar.setObjectName("menubar")
        self.menuTilda_MainWindow = QtWidgets.QMenu(self.menubar)
        self.menuTilda_MainWindow.setObjectName("menuTilda_MainWindow")
        self.menuView = QtWidgets.QMenu(self.menubar)
        self.menuView.setObjectName("menuView")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        self.menuConfigure = QtWidgets.QMenu(self.menubar)
        self.menuConfigure.setObjectName("menuConfigure")
        TildaMainWindow.setMenuBar(self.menubar)
        self.actionTracks = QtWidgets.QAction(TildaMainWindow)
        self.actionTracks.setObjectName("actionTracks")
        self.actionScan_Control = QtWidgets.QAction(TildaMainWindow)
        self.actionScan_Control.setObjectName("actionScan_Control")
        self.actionVersion = QtWidgets.QAction(TildaMainWindow)
        self.actionVersion.setObjectName("actionVersion")
        self.actionHardware_setup = QtWidgets.QAction(TildaMainWindow)
        self.actionHardware_setup.setObjectName("actionHardware_setup")
        self.actionWorking_directory = QtWidgets.QAction(TildaMainWindow)
        self.actionWorking_directory.setObjectName("actionWorking_directory")
        self.actionVoltage_Measurement = QtWidgets.QAction(TildaMainWindow)
        self.actionVoltage_Measurement.setObjectName("actionVoltage_Measurement")
        self.menuTilda_MainWindow.addAction(self.actionWorking_directory)
        self.menuView.addAction(self.actionTracks)
        self.menuView.addAction(self.actionScan_Control)
        self.menuHelp.addAction(self.actionVersion)
        self.menuConfigure.addAction(self.actionVoltage_Measurement)
        self.menubar.addAction(self.menuTilda_MainWindow.menuAction())
        self.menubar.addAction(self.menuView.menuAction())
        self.menubar.addAction(self.menuConfigure.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(TildaMainWindow)
        QtCore.QMetaObject.connectSlotsByName(TildaMainWindow)

    def retranslateUi(self, TildaMainWindow):
        _translate = QtCore.QCoreApplication.translate
        TildaMainWindow.setWindowTitle(_translate("TildaMainWindow", "Tilda"))
        self.label_workdir.setText(_translate("TildaMainWindow", "Working directory:"))
        self.label_workdir_set.setText(_translate("TildaMainWindow", "please choose a working directory in File->working directory"))
        self.menuTilda_MainWindow.setTitle(_translate("TildaMainWindow", "File"))
        self.menuView.setTitle(_translate("TildaMainWindow", "Tools"))
        self.menuHelp.setTitle(_translate("TildaMainWindow", "Help"))
        self.menuConfigure.setTitle(_translate("TildaMainWindow", "Configure"))
        self.actionTracks.setText(_translate("TildaMainWindow", "Tracks"))
        self.actionScan_Control.setText(_translate("TildaMainWindow", "Scan Control"))
        self.actionVersion.setText(_translate("TildaMainWindow", "Version"))
        self.actionHardware_setup.setText(_translate("TildaMainWindow", "hardware setup"))
        self.actionWorking_directory.setText(_translate("TildaMainWindow", "working directory"))
        self.actionVoltage_Measurement.setText(_translate("TildaMainWindow", "Voltage Measurement"))

