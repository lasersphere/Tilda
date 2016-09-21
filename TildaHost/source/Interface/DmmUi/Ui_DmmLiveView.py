# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_DmmLiveView.ui'
#
# Created by: PyQt5 UI code generator 5.5
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(450, 293)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.tabWidget.setObjectName("tabWidget")
        self.tab_0 = QtWidgets.QWidget()
        self.tab_0.setObjectName("tab_0")
        self.tabWidget.addTab(self.tab_0, "")
        self.verticalLayout.addWidget(self.tabWidget)
        self.formLayout_pulse_len_and_timeout = QtWidgets.QFormLayout()
        self.formLayout_pulse_len_and_timeout.setObjectName("formLayout_pulse_len_and_timeout")
        self.label_measVoltPulseLength_mu_s = QtWidgets.QLabel(self.centralwidget)
        self.label_measVoltPulseLength_mu_s.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.label_measVoltPulseLength_mu_s.setObjectName("label_measVoltPulseLength_mu_s")
        self.formLayout_pulse_len_and_timeout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_measVoltPulseLength_mu_s)
        self.doubleSpinBox_measVoltPulseLength_mu_s = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_measVoltPulseLength_mu_s.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.doubleSpinBox_measVoltPulseLength_mu_s.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.doubleSpinBox_measVoltPulseLength_mu_s.setKeyboardTracking(False)
        self.doubleSpinBox_measVoltPulseLength_mu_s.setDecimals(3)
        self.doubleSpinBox_measVoltPulseLength_mu_s.setMaximum(107374182.0)
        self.doubleSpinBox_measVoltPulseLength_mu_s.setObjectName("doubleSpinBox_measVoltPulseLength_mu_s")
        self.formLayout_pulse_len_and_timeout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.doubleSpinBox_measVoltPulseLength_mu_s)
        self.doubleSpinBox_measVoltTimeout_mu_s_set = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_measVoltTimeout_mu_s_set.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.doubleSpinBox_measVoltTimeout_mu_s_set.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.doubleSpinBox_measVoltTimeout_mu_s_set.setKeyboardTracking(False)
        self.doubleSpinBox_measVoltTimeout_mu_s_set.setDecimals(3)
        self.doubleSpinBox_measVoltTimeout_mu_s_set.setMaximum(42949.0)
        self.doubleSpinBox_measVoltTimeout_mu_s_set.setObjectName("doubleSpinBox_measVoltTimeout_mu_s_set")
        self.formLayout_pulse_len_and_timeout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.doubleSpinBox_measVoltTimeout_mu_s_set)
        self.label_measVoltTimeout_mu_s = QtWidgets.QLabel(self.centralwidget)
        self.label_measVoltTimeout_mu_s.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.label_measVoltTimeout_mu_s.setObjectName("label_measVoltTimeout_mu_s")
        self.formLayout_pulse_len_and_timeout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_measVoltTimeout_mu_s)
        self.verticalLayout.addLayout(self.formLayout_pulse_len_and_timeout)
        self.pushButton_confirm = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_confirm.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.pushButton_confirm.setAutoDefault(True)
        self.pushButton_confirm.setDefault(True)
        self.pushButton_confirm.setObjectName("pushButton_confirm")
        self.verticalLayout.addWidget(self.pushButton_confirm)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_0), _translate("MainWindow", "Tab 1"))
        self.label_measVoltPulseLength_mu_s.setText(_translate("MainWindow", "Pulse length for voltage measurement request [µs]"))
        self.label_measVoltTimeout_mu_s.setText(_translate("MainWindow", "timeout for voltage measurement [ms]"))
        self.pushButton_confirm.setText(_translate("MainWindow", "ok"))

