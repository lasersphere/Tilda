# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_PreScanMain.ui'
#
# Created by: PyQt5 UI code generator 5.5
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_PreScanMainWin(object):
    def setupUi(self, PreScanMainWin):
        PreScanMainWin.setObjectName("PreScanMainWin")
        PreScanMainWin.resize(464, 600)
        self.centralwidget = QtWidgets.QWidget(PreScanMainWin)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setObjectName("comboBox")
        self.verticalLayout.addWidget(self.comboBox)
        self.mainTabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.mainTabWidget.setObjectName("mainTabWidget")
        self.volt_tab = QtWidgets.QWidget()
        self.volt_tab.setObjectName("volt_tab")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.volt_tab)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.voltage_mainwidget = QtWidgets.QWidget(self.volt_tab)
        self.voltage_mainwidget.setObjectName("voltage_mainwidget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.voltage_mainwidget)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.checkBox_voltage_measure = QtWidgets.QCheckBox(self.voltage_mainwidget)
        self.checkBox_voltage_measure.setObjectName("checkBox_voltage_measure")
        self.verticalLayout_3.addWidget(self.checkBox_voltage_measure)
        self.tabWidget = QtWidgets.QTabWidget(self.voltage_mainwidget)
        self.tabWidget.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.tabWidget.setObjectName("tabWidget")
        self.tab_0 = QtWidgets.QWidget()
        self.tab_0.setObjectName("tab_0")
        self.tabWidget.addTab(self.tab_0, "")
        self.verticalLayout_3.addWidget(self.tabWidget)
        self.verticalLayout_2.addWidget(self.voltage_mainwidget)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.label_measVoltPulseLength_mu_s = QtWidgets.QLabel(self.volt_tab)
        self.label_measVoltPulseLength_mu_s.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.label_measVoltPulseLength_mu_s.setObjectName("label_measVoltPulseLength_mu_s")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_measVoltPulseLength_mu_s)
        self.label_measVoltTimeout_mu_s = QtWidgets.QLabel(self.volt_tab)
        self.label_measVoltTimeout_mu_s.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.label_measVoltTimeout_mu_s.setObjectName("label_measVoltTimeout_mu_s")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_measVoltTimeout_mu_s)
        self.doubleSpinBox_measVoltTimeout_mu_s_set = QtWidgets.QDoubleSpinBox(self.volt_tab)
        self.doubleSpinBox_measVoltTimeout_mu_s_set.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.doubleSpinBox_measVoltTimeout_mu_s_set.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.doubleSpinBox_measVoltTimeout_mu_s_set.setKeyboardTracking(False)
        self.doubleSpinBox_measVoltTimeout_mu_s_set.setDecimals(3)
        self.doubleSpinBox_measVoltTimeout_mu_s_set.setMaximum(42949.0)
        self.doubleSpinBox_measVoltTimeout_mu_s_set.setObjectName("doubleSpinBox_measVoltTimeout_mu_s_set")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.doubleSpinBox_measVoltTimeout_mu_s_set)
        self.doubleSpinBox_measVoltPulseLength_mu_s = QtWidgets.QDoubleSpinBox(self.volt_tab)
        self.doubleSpinBox_measVoltPulseLength_mu_s.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.doubleSpinBox_measVoltPulseLength_mu_s.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.doubleSpinBox_measVoltPulseLength_mu_s.setKeyboardTracking(False)
        self.doubleSpinBox_measVoltPulseLength_mu_s.setDecimals(3)
        self.doubleSpinBox_measVoltPulseLength_mu_s.setMaximum(107374182.0)
        self.doubleSpinBox_measVoltPulseLength_mu_s.setObjectName("doubleSpinBox_measVoltPulseLength_mu_s")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.doubleSpinBox_measVoltPulseLength_mu_s)
        self.verticalLayout_2.addLayout(self.formLayout)
        self.mainTabWidget.addTab(self.volt_tab, "")
        self.triton_tab = QtWidgets.QWidget()
        self.triton_tab.setObjectName("triton_tab")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.triton_tab)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.checkBox_triton_measure = QtWidgets.QCheckBox(self.triton_tab)
        self.checkBox_triton_measure.setObjectName("checkBox_triton_measure")
        self.verticalLayout_4.addWidget(self.checkBox_triton_measure)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.triton_tab)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.label_channels = QtWidgets.QLabel(self.triton_tab)
        self.label_channels.setObjectName("label_channels")
        self.gridLayout.addWidget(self.label_channels, 0, 1, 1, 1)
        self.listWidget_devices = QtWidgets.QListWidget(self.triton_tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.listWidget_devices.sizePolicy().hasHeightForWidth())
        self.listWidget_devices.setSizePolicy(sizePolicy)
        self.listWidget_devices.setObjectName("listWidget_devices")
        self.gridLayout.addWidget(self.listWidget_devices, 1, 0, 1, 1)
        self.tableWidget_channels = QtWidgets.QTableWidget(self.triton_tab)
        self.tableWidget_channels.setObjectName("tableWidget_channels")
        self.tableWidget_channels.setColumnCount(0)
        self.tableWidget_channels.setRowCount(0)
        self.gridLayout.addWidget(self.tableWidget_channels, 1, 1, 1, 1)
        self.verticalLayout_4.addLayout(self.gridLayout)
        self.mainTabWidget.addTab(self.triton_tab, "")
        self.verticalLayout.addWidget(self.mainTabWidget)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_wait_after_switchbox = QtWidgets.QLabel(self.centralwidget)
        self.label_wait_after_switchbox.setObjectName("label_wait_after_switchbox")
        self.horizontalLayout.addWidget(self.label_wait_after_switchbox)
        self.doubleSpinBox_wait_after_switchbox = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_wait_after_switchbox.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.doubleSpinBox_wait_after_switchbox.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.doubleSpinBox_wait_after_switchbox.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.doubleSpinBox_wait_after_switchbox.setKeyboardTracking(False)
        self.doubleSpinBox_wait_after_switchbox.setMaximum(10.0)
        self.doubleSpinBox_wait_after_switchbox.setObjectName("doubleSpinBox_wait_after_switchbox")
        self.horizontalLayout.addWidget(self.doubleSpinBox_wait_after_switchbox)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.buttonBox = QtWidgets.QDialogButtonBox(self.centralwidget)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)
        PreScanMainWin.setCentralWidget(self.centralwidget)

        self.retranslateUi(PreScanMainWin)
        self.mainTabWidget.setCurrentIndex(1)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(PreScanMainWin)

    def retranslateUi(self, PreScanMainWin):
        _translate = QtCore.QCoreApplication.translate
        PreScanMainWin.setWindowTitle(_translate("PreScanMainWin", "MainWindow"))
        self.checkBox_voltage_measure.setText(_translate("PreScanMainWin", "measure"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_0), _translate("PreScanMainWin", "Tab 1"))
        self.label_measVoltPulseLength_mu_s.setText(_translate("PreScanMainWin", "Pulse length for voltage measurement request [µs]"))
        self.label_measVoltTimeout_mu_s.setText(_translate("PreScanMainWin", "timeout for voltage measurement [ms]"))
        self.mainTabWidget.setTabText(self.mainTabWidget.indexOf(self.volt_tab), _translate("PreScanMainWin", "voltage meas."))
        self.checkBox_triton_measure.setText(_translate("PreScanMainWin", "measure"))
        self.label.setText(_translate("PreScanMainWin", "devices"))
        self.label_channels.setText(_translate("PreScanMainWin", "channels"))
        self.mainTabWidget.setTabText(self.mainTabWidget.indexOf(self.triton_tab), _translate("PreScanMainWin", "triton"))
        self.label_wait_after_switchbox.setText(_translate("PreScanMainWin", "wait after switchbox changed [s]"))

