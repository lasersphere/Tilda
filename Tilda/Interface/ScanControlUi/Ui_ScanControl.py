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
        MainWindowScanControl.resize(376, 226)
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
        self.checkBox_reps_as_go = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_reps_as_go.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.checkBox_reps_as_go.setObjectName("checkBox_reps_as_go")
        self.verticalLayout.addWidget(self.checkBox_reps_as_go)
        MainWindowScanControl.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindowScanControl)
        self.statusbar.setObjectName("statusbar")
        MainWindowScanControl.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(MainWindowScanControl)
        self.toolBar.setObjectName("toolBar")
        MainWindowScanControl.addToolBar(QtCore.Qt.LeftToolBarArea, self.toolBar)
        self.actionErgo = QtWidgets.QAction(MainWindowScanControl)
        self.actionErgo.setObjectName("actionErgo")
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
        self.actionGo_on_file = QtWidgets.QAction(MainWindowScanControl)
        self.actionGo_on_file.setObjectName("actionGo_on_file")
        self.actionRe_open_plot_win = QtWidgets.QAction(MainWindowScanControl)
        self.actionRe_open_plot_win.setObjectName("actionRe_open_plot_win")
        self.toolBar.addAction(self.actionErgo)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionGo_on_file)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionSetup_Isotope)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionAdd_Track)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.action_remove_track)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionSave_settings_to_database)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionRe_open_plot_win)

        self.retranslateUi(MainWindowScanControl)
        QtCore.QMetaObject.connectSlotsByName(MainWindowScanControl)

    def retranslateUi(self, MainWindowScanControl):
        _translate = QtCore.QCoreApplication.translate
        MainWindowScanControl.setWindowTitle(_translate("MainWindowScanControl", "Scan - undefined"))
        self.label.setText(_translate("MainWindowScanControl", "# of repetitions:"))
        self.spinBox_num_of_reps.setToolTip(_translate("MainWindowScanControl", "<html><head/><body><p>each repetition will be a seperate file</p></body></html>"))
        self.checkBox_reps_as_go.setText(_translate("MainWindowScanControl", "repetitions as go"))
        self.toolBar.setWindowTitle(_translate("MainWindowScanControl", "toolBar"))
        self.actionErgo.setText(_translate("MainWindowScanControl", "ergo"))
        self.actionErgo.setToolTip(_translate("MainWindowScanControl", "Starts a new measurement, as configured in the settings, Ctrl+G"))
        self.actionErgo.setShortcut(_translate("MainWindowScanControl", "Ctrl+E"))
        self.actionSetup_Isotope.setText(_translate("MainWindowScanControl", "setup Isotope"))
        self.actionSetup_Isotope.setShortcut(_translate("MainWindowScanControl", "Ctrl+I"))
        self.actionAdd_Track.setText(_translate("MainWindowScanControl", "add track"))
        self.actionSave_settings_to_database.setText(_translate("MainWindowScanControl", "save settings to database"))
        self.actionSave_settings_to_database.setShortcut(_translate("MainWindowScanControl", "Ctrl+S"))
        self.action_remove_track.setText(_translate("MainWindowScanControl", "remove track"))
        self.actionGo_on_file.setText(_translate("MainWindowScanControl", "go on file"))
        self.actionGo_on_file.setToolTip(_translate("MainWindowScanControl", "continue running on an existing file with the same settings as in the file"))
        self.actionGo_on_file.setShortcut(_translate("MainWindowScanControl", "Ctrl+G"))
        self.actionRe_open_plot_win.setText(_translate("MainWindowScanControl", "re open plot win"))

