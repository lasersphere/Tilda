# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_Main.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_TildaMainWindow(object):
    def setupUi(self, TildaMainWindow):
        TildaMainWindow.setObjectName("TildaMainWindow")
        TildaMainWindow.resize(548, 252)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("../icons/Tilda256.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        TildaMainWindow.setWindowIcon(icon)
        TildaMainWindow.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.centralwidget = QtWidgets.QWidget(TildaMainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 5, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 3, 0, 1, 1)
        self.label_main_status = QtWidgets.QLabel(self.centralwidget)
        self.label_main_status.setObjectName("label_main_status")
        self.gridLayout.addWidget(self.label_main_status, 3, 2, 1, 1)
        self.label_workdir_set = QtWidgets.QLabel(self.centralwidget)
        self.label_workdir_set.setObjectName("label_workdir_set")
        self.gridLayout.addWidget(self.label_workdir_set, 0, 2, 1, 1)
        self.label_database = QtWidgets.QLabel(self.centralwidget)
        self.label_database.setObjectName("label_database")
        self.gridLayout.addWidget(self.label_database, 1, 2, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.label_acc_volt_set = QtWidgets.QLabel(self.centralwidget)
        self.label_acc_volt_set.setObjectName("label_acc_volt_set")
        self.gridLayout.addWidget(self.label_acc_volt_set, 8, 2, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 8, 0, 1, 1)
        self.label_laser_freq_set = QtWidgets.QLabel(self.centralwidget)
        self.label_laser_freq_set.setObjectName("label_laser_freq_set")
        self.gridLayout.addWidget(self.label_laser_freq_set, 7, 2, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 7, 0, 1, 1)
        self.label_sequencer_status_set = QtWidgets.QLabel(self.centralwidget)
        self.label_sequencer_status_set.setObjectName("label_sequencer_status_set")
        self.gridLayout.addWidget(self.label_sequencer_status_set, 5, 2, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 6, 0, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 4, 0, 1, 1)
        self.label_fpga_state_set = QtWidgets.QLabel(self.centralwidget)
        self.label_fpga_state_set.setObjectName("label_fpga_state_set")
        self.gridLayout.addWidget(self.label_fpga_state_set, 4, 2, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 11, 0, 1, 1)
        self.label_dmm_status = QtWidgets.QLabel(self.centralwidget)
        self.label_dmm_status.setObjectName("label_dmm_status")
        self.gridLayout.addWidget(self.label_dmm_status, 6, 2, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 10, 0, 1, 1)
        self.label_workdir = QtWidgets.QLabel(self.centralwidget)
        self.label_workdir.setObjectName("label_workdir")
        self.gridLayout.addWidget(self.label_workdir, 0, 0, 1, 1)
        self.pushButton_open_dir = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_open_dir.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedKingdom))
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("../icons/open_folder.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_open_dir.setIcon(icon1)
        self.pushButton_open_dir.setObjectName("pushButton_open_dir")
        self.gridLayout.addWidget(self.pushButton_open_dir, 0, 3, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setObjectName("label_9")
        self.gridLayout.addWidget(self.label_9, 9, 0, 1, 1)
        self.label_triton_subscription = QtWidgets.QLabel(self.centralwidget)
        self.label_triton_subscription.setObjectName("label_triton_subscription")
        self.gridLayout.addWidget(self.label_triton_subscription, 9, 2, 1, 1)
        TildaMainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(TildaMainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 548, 21))
        self.menubar.setObjectName("menubar")
        self.menuTilda_MainWindow = QtWidgets.QMenu(self.menubar)
        self.menuTilda_MainWindow.setObjectName("menuTilda_MainWindow")
        self.menuView = QtWidgets.QMenu(self.menubar)
        self.menuView.setObjectName("menuView")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        self.menuConfigure = QtWidgets.QMenu(self.menubar)
        self.menuConfigure.setObjectName("menuConfigure")
        self.menuAnalysis = QtWidgets.QMenu(self.menubar)
        self.menuAnalysis.setObjectName("menuAnalysis")
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
        self.actionPost_acceleration_power_supply_control = QtWidgets.QAction(TildaMainWindow)
        self.actionPost_acceleration_power_supply_control.setObjectName("actionPost_acceleration_power_supply_control")
        self.actionSimple_Counter = QtWidgets.QAction(TildaMainWindow)
        self.actionSimple_Counter.setCheckable(False)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("../icons/SimpleCounterNew.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionSimple_Counter.setIcon(icon2)
        font = QtGui.QFont()
        self.actionSimple_Counter.setFont(font)
        self.actionSimple_Counter.setObjectName("actionSimple_Counter")
        self.actionSet_Laser_Frequency = QtWidgets.QAction(TildaMainWindow)
        self.actionSet_Laser_Frequency.setObjectName("actionSet_Laser_Frequency")
        self.actionSet_acceleration_voltage = QtWidgets.QAction(TildaMainWindow)
        self.actionSet_acceleration_voltage.setObjectName("actionSet_acceleration_voltage")
        self.actionTilda_Passive = QtWidgets.QAction(TildaMainWindow)
        self.actionTilda_Passive.setObjectName("actionTilda_Passive")
        self.actionLoad_spectra = QtWidgets.QAction(TildaMainWindow)
        self.actionLoad_spectra.setObjectName("actionLoad_spectra")
        self.actionDigital_Multimeters = QtWidgets.QAction(TildaMainWindow)
        self.actionDigital_Multimeters.setObjectName("actionDigital_Multimeters")
        self.actionPolliFit = QtWidgets.QAction(TildaMainWindow)
        self.actionPolliFit.setObjectName("actionPolliFit")
        self.actionPulse_pattern_generator = QtWidgets.QAction(TildaMainWindow)
        self.actionPulse_pattern_generator.setObjectName("actionPulse_pattern_generator")
        self.actionShow_scan_finished_win = QtWidgets.QAction(TildaMainWindow)
        self.actionShow_scan_finished_win.setCheckable(True)
        self.actionShow_scan_finished_win.setChecked(True)
        self.actionShow_scan_finished_win.setObjectName("actionShow_scan_finished_win")
        self.actionPre_scan_timeout = QtWidgets.QAction(TildaMainWindow)
        self.actionPre_scan_timeout.setObjectName("actionPre_scan_timeout")
        self.actionJob_Stacker = QtWidgets.QAction(TildaMainWindow)
        self.actionJob_Stacker.setObjectName("actionJob_Stacker")
        self.actionoptions = QtWidgets.QAction(TildaMainWindow)
        self.actionoptions.setObjectName("actionoptions")
        self.menuTilda_MainWindow.addAction(self.actionWorking_directory)
        self.menuTilda_MainWindow.addAction(self.actionLoad_spectra)
        self.menuView.addAction(self.actionScan_Control)
        self.menuView.addAction(self.actionSimple_Counter)
        self.menuView.addAction(self.actionSet_Laser_Frequency)
        self.menuView.addAction(self.actionSet_acceleration_voltage)
        self.menuView.addAction(self.actionPulse_pattern_generator)
        self.menuView.addAction(self.actionPost_acceleration_power_supply_control)
        self.menuView.addAction(self.actionDigital_Multimeters)
        self.menuView.addAction(self.actionJob_Stacker)
        self.menuHelp.addAction(self.actionVersion)
        self.menuConfigure.addAction(self.actionoptions)
        self.menuAnalysis.addAction(self.actionPolliFit)
        self.menubar.addAction(self.menuTilda_MainWindow.menuAction())
        self.menubar.addAction(self.menuView.menuAction())
        self.menubar.addAction(self.menuConfigure.menuAction())
        self.menubar.addAction(self.menuAnalysis.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(TildaMainWindow)
        QtCore.QMetaObject.connectSlotsByName(TildaMainWindow)

    def retranslateUi(self, TildaMainWindow):
        _translate = QtCore.QCoreApplication.translate
        TildaMainWindow.setWindowTitle(_translate("TildaMainWindow", "Tilda"))
        self.label_6.setText(_translate("TildaMainWindow", "sequencer status:"))
        self.label.setText(_translate("TildaMainWindow", "main state:"))
        self.label_main_status.setText(_translate("TildaMainWindow", "TextLabel"))
        self.label_workdir_set.setText(_translate("TildaMainWindow", "please choose a working directory in File->working directory"))
        self.label_database.setText(_translate("TildaMainWindow", "TextLabel"))
        self.label_2.setText(_translate("TildaMainWindow", "database:"))
        self.label_acc_volt_set.setText(_translate("TildaMainWindow", "TextLabel"))
        self.label_4.setText(_translate("TildaMainWindow", "acceleration voltage [V]*:"))
        self.label_laser_freq_set.setText(_translate("TildaMainWindow", "TextLabel"))
        self.label_3.setText(_translate("TildaMainWindow", "laser frequency [cm-1]*:"))
        self.label_sequencer_status_set.setText(_translate("TildaMainWindow", "TextLabel"))
        self.label_8.setText(_translate("TildaMainWindow", "digital multimeter status*:"))
        self.label_7.setText(_translate("TildaMainWindow", "fpga status:"))
        self.label_fpga_state_set.setText(_translate("TildaMainWindow", "TextLabel"))
        self.label_5.setText(_translate("TildaMainWindow", "*double click on value to change"))
        self.label_dmm_status.setText(_translate("TildaMainWindow", "TextLabel"))
        self.label_workdir.setText(_translate("TildaMainWindow", "Working directory*:"))
        self.pushButton_open_dir.setText(_translate("TildaMainWindow", "open"))
        self.label_9.setText(_translate("TildaMainWindow", "triton subscriptions*:"))
        self.label_triton_subscription.setText(_translate("TildaMainWindow", "TextLabel"))
        self.menuTilda_MainWindow.setTitle(_translate("TildaMainWindow", "File"))
        self.menuView.setTitle(_translate("TildaMainWindow", "Tools"))
        self.menuHelp.setTitle(_translate("TildaMainWindow", "Help"))
        self.menuConfigure.setTitle(_translate("TildaMainWindow", "Configure"))
        self.menuAnalysis.setTitle(_translate("TildaMainWindow", "Analysis"))
        self.actionTracks.setText(_translate("TildaMainWindow", "Tracks"))
        self.actionScan_Control.setText(_translate("TildaMainWindow", "Scan Control"))
        self.actionScan_Control.setShortcut(_translate("TildaMainWindow", "Ctrl+S"))
        self.actionVersion.setText(_translate("TildaMainWindow", "Version"))
        self.actionHardware_setup.setText(_translate("TildaMainWindow", "hardware setup"))
        self.actionWorking_directory.setText(_translate("TildaMainWindow", "working directory"))
        self.actionWorking_directory.setToolTip(_translate("TildaMainWindow", "working directory, Ctrl+W"))
        self.actionWorking_directory.setShortcut(_translate("TildaMainWindow", "Ctrl+W"))
        self.actionVoltage_Measurement.setText(_translate("TildaMainWindow", "Voltage Measurement"))
        self.actionPost_acceleration_power_supply_control.setText(_translate("TildaMainWindow", "Post acceleration power supplies"))
        self.actionPost_acceleration_power_supply_control.setToolTip(_translate("TildaMainWindow", "Post acceleration power supplies, Ctrl+P"))
        self.actionPost_acceleration_power_supply_control.setShortcut(_translate("TildaMainWindow", "Ctrl+P"))
        self.actionSimple_Counter.setText(_translate("TildaMainWindow", "Simple Counter"))
        self.actionSet_Laser_Frequency.setText(_translate("TildaMainWindow", "set laser frequency"))
        self.actionSet_Laser_Frequency.setToolTip(_translate("TildaMainWindow", "set laser frequency, Ctrl+L"))
        self.actionSet_Laser_Frequency.setShortcut(_translate("TildaMainWindow", "Ctrl+L"))
        self.actionSet_acceleration_voltage.setText(_translate("TildaMainWindow", "set acceleration voltage"))
        self.actionSet_acceleration_voltage.setToolTip(_translate("TildaMainWindow", "set acceleration voltage, Ctrl+V"))
        self.actionSet_acceleration_voltage.setShortcut(_translate("TildaMainWindow", "Ctrl+V"))
        self.actionTilda_Passive.setText(_translate("TildaMainWindow", "Tilda Passive"))
        self.actionLoad_spectra.setText(_translate("TildaMainWindow", "load spectra"))
        self.actionLoad_spectra.setToolTip(_translate("TildaMainWindow", "load spectra, Ctrl+O"))
        self.actionLoad_spectra.setShortcut(_translate("TildaMainWindow", "Ctrl+O"))
        self.actionDigital_Multimeters.setText(_translate("TildaMainWindow", "Digital Multimeters"))
        self.actionDigital_Multimeters.setToolTip(_translate("TildaMainWindow", "Digital Multimeters, Ctrl+D"))
        self.actionDigital_Multimeters.setShortcut(_translate("TildaMainWindow", "Ctrl+D"))
        self.actionPolliFit.setText(_translate("TildaMainWindow", "PolliFit"))
        self.actionPolliFit.setToolTip(_translate("TildaMainWindow", "PolliFit, Ctrl+A"))
        self.actionPolliFit.setShortcut(_translate("TildaMainWindow", "Ctrl+A"))
        self.actionPulse_pattern_generator.setText(_translate("TildaMainWindow", "pulse pattern generator"))
        self.actionPulse_pattern_generator.setShortcut(_translate("TildaMainWindow", "Ctrl+G"))
        self.actionShow_scan_finished_win.setText(_translate("TildaMainWindow", "show scan finished win"))
        self.actionPre_scan_timeout.setText(_translate("TildaMainWindow", "pre scan timeout"))
        self.actionJob_Stacker.setText(_translate("TildaMainWindow", "Job Stacker"))
        self.actionJob_Stacker.setToolTip(_translate("TildaMainWindow", "Job Stacker, Ctrl+J"))
        self.actionJob_Stacker.setShortcut(_translate("TildaMainWindow", "Ctrl+J"))
        self.actionoptions.setText(_translate("TildaMainWindow", "Options"))
