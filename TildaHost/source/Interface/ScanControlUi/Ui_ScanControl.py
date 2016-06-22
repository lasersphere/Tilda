# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_ScanControl.ui'
#
# Created by: PyQt5 UI code generator 5.5
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindowScanControl(object):
    def setupUi(self, MainWindowScanControl):
        MainWindowScanControl.setObjectName("MainWindowScanControl")
        MainWindowScanControl.resize(341, 211)
        MainWindowScanControl.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        MainWindowScanControl.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.centralwidget = QtWidgets.QWidget(MainWindowScanControl)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.listWidget = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget.setObjectName("listWidget")
        self.verticalLayout.addWidget(self.listWidget)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.spinBox_num_of_reps = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox_num_of_reps.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.spinBox_num_of_reps.setMaximum(10000)
        self.spinBox_num_of_reps.setSingleStep(1)
        self.spinBox_num_of_reps.setProperty("value", 1)
        self.spinBox_num_of_reps.setObjectName("spinBox_num_of_reps")
        self.horizontalLayout.addWidget(self.spinBox_num_of_reps)
        self.verticalLayout.addLayout(self.horizontalLayout)
        MainWindowScanControl.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindowScanControl)
        self.statusbar.setObjectName("statusbar")
        MainWindowScanControl.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(MainWindowScanControl)
        self.toolBar.setObjectName("toolBar")
        MainWindowScanControl.addToolBar(QtCore.Qt.LeftToolBarArea, self.toolBar)
        self.actionGo = QtWidgets.QAction(MainWindowScanControl)
        self.actionGo.setObjectName("actionGo")
        self.actionSetup_Isotope = QtWidgets.QAction(MainWindowScanControl)
        self.actionSetup_Isotope.setObjectName("actionSetup_Isotope")
        self.actionAdd_Track = QtWidgets.QAction(MainWindowScanControl)
        self.actionAdd_Track.setObjectName("actionAdd_Track")
        self.actionSave_settings_to_database = QtWidgets.QAction(MainWindowScanControl)
        icon = QtGui.QIcon.fromTheme("save")
        self.actionSave_settings_to_database.setIcon(icon)
        self.actionSave_settings_to_database.setObjectName("actionSave_settings_to_database")
        self.action_remove_track = QtWidgets.QAction(MainWindowScanControl)
        self.action_remove_track.setObjectName("action_remove_track")
        self.actionConfigure_voltage_measurement = QtWidgets.QAction(MainWindowScanControl)
        self.actionConfigure_voltage_measurement.setObjectName("actionConfigure_voltage_measurement")
        self.toolBar.addAction(self.actionGo)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionSetup_Isotope)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionAdd_Track)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.action_remove_track)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionConfigure_voltage_measurement)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionSave_settings_to_database)

        self.retranslateUi(MainWindowScanControl)
        QtCore.QMetaObject.connectSlotsByName(MainWindowScanControl)

    def retranslateUi(self, MainWindowScanControl):
        _translate = QtCore.QCoreApplication.translate
        MainWindowScanControl.setWindowTitle(_translate("MainWindowScanControl", "Scan - undefined"))
        self.label.setText(_translate("MainWindowScanControl", "# of repetitions:"))
        self.toolBar.setWindowTitle(_translate("MainWindowScanControl", "toolBar"))
        self.actionGo.setText(_translate("MainWindowScanControl", "Go"))
        self.actionGo.setToolTip(_translate("MainWindowScanControl", "Starts the measurement, Ctrl+G"))
        self.actionGo.setShortcut(_translate("MainWindowScanControl", "Ctrl+G"))
        self.actionSetup_Isotope.setText(_translate("MainWindowScanControl", "setup Isotope"))
        self.actionSetup_Isotope.setShortcut(_translate("MainWindowScanControl", "Ctrl+I"))
        self.actionAdd_Track.setText(_translate("MainWindowScanControl", "add track"))
        self.actionSave_settings_to_database.setText(_translate("MainWindowScanControl", "save settings to database"))
        self.actionSave_settings_to_database.setShortcut(_translate("MainWindowScanControl", "Ctrl+S"))
        self.action_remove_track.setText(_translate("MainWindowScanControl", "remove track"))
        self.actionConfigure_voltage_measurement.setText(_translate("MainWindowScanControl", "configure voltage measurement"))

