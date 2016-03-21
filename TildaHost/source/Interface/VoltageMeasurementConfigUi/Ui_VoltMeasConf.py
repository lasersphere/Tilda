# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_VoltMeasConf.ui'
#
# Created by: PyQt5 UI code generator 5.5
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_VoltMeasConfMainWin(object):
    def setupUi(self, VoltMeasConfMainWin):
        VoltMeasConfMainWin.setObjectName("VoltMeasConfMainWin")
        VoltMeasConfMainWin.resize(410, 70)
        VoltMeasConfMainWin.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.centralwidget = QtWidgets.QWidget(VoltMeasConfMainWin)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.label_measVoltTimeout_mu_s_set = QtWidgets.QLabel(self.centralwidget)
        self.label_measVoltTimeout_mu_s_set.setObjectName("label_measVoltTimeout_mu_s_set")
        self.gridLayout.addWidget(self.label_measVoltTimeout_mu_s_set, 1, 2, 1, 1)
        self.label_measVoltPulseLength_mu_s = QtWidgets.QLabel(self.centralwidget)
        self.label_measVoltPulseLength_mu_s.setObjectName("label_measVoltPulseLength_mu_s")
        self.gridLayout.addWidget(self.label_measVoltPulseLength_mu_s, 0, 0, 1, 1)
        self.doubleSpinBox_measVoltPulseLength_mu_s = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_measVoltPulseLength_mu_s.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.doubleSpinBox_measVoltPulseLength_mu_s.setKeyboardTracking(False)
        self.doubleSpinBox_measVoltPulseLength_mu_s.setDecimals(3)
        self.doubleSpinBox_measVoltPulseLength_mu_s.setMaximum(1000000.0)
        self.doubleSpinBox_measVoltPulseLength_mu_s.setObjectName("doubleSpinBox_measVoltPulseLength_mu_s")
        self.gridLayout.addWidget(self.doubleSpinBox_measVoltPulseLength_mu_s, 0, 1, 1, 1)
        self.label_measVoltPulseLength_mu_s_set = QtWidgets.QLabel(self.centralwidget)
        self.label_measVoltPulseLength_mu_s_set.setObjectName("label_measVoltPulseLength_mu_s_set")
        self.gridLayout.addWidget(self.label_measVoltPulseLength_mu_s_set, 0, 2, 1, 1)
        self.label_measVoltTimeout_mu_s = QtWidgets.QLabel(self.centralwidget)
        self.label_measVoltTimeout_mu_s.setObjectName("label_measVoltTimeout_mu_s")
        self.gridLayout.addWidget(self.label_measVoltTimeout_mu_s, 1, 0, 1, 1)
        self.doubleSpinBox_measVoltTimeout_mu_s_set = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_measVoltTimeout_mu_s_set.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.doubleSpinBox_measVoltTimeout_mu_s_set.setKeyboardTracking(False)
        self.doubleSpinBox_measVoltTimeout_mu_s_set.setDecimals(3)
        self.doubleSpinBox_measVoltTimeout_mu_s_set.setMaximum(1000000.0)
        self.doubleSpinBox_measVoltTimeout_mu_s_set.setObjectName("doubleSpinBox_measVoltTimeout_mu_s_set")
        self.gridLayout.addWidget(self.doubleSpinBox_measVoltTimeout_mu_s_set, 1, 1, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 2, 1, 1, 1)
        VoltMeasConfMainWin.setCentralWidget(self.centralwidget)

        self.retranslateUi(VoltMeasConfMainWin)
        QtCore.QMetaObject.connectSlotsByName(VoltMeasConfMainWin)

    def retranslateUi(self, VoltMeasConfMainWin):
        _translate = QtCore.QCoreApplication.translate
        VoltMeasConfMainWin.setWindowTitle(_translate("VoltMeasConfMainWin", "Voltage Measurement Configuration"))
        self.label_measVoltTimeout_mu_s_set.setText(_translate("VoltMeasConfMainWin", "0"))
        self.label_measVoltPulseLength_mu_s.setText(_translate("VoltMeasConfMainWin", "Pulse length for voltage measurement request"))
        self.label_measVoltPulseLength_mu_s_set.setText(_translate("VoltMeasConfMainWin", "0"))
        self.label_measVoltTimeout_mu_s.setText(_translate("VoltMeasConfMainWin", "timeout for voltage measurement"))

