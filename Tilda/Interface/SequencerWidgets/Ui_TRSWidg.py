# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_TRSWidg.ui'
#
# Created by: PyQt5 UI code generator 5.5
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_TRSWidg(object):
    def setupUi(self, TRSWidg):
        TRSWidg.setObjectName("TRSWidg")
        TRSWidg.resize(358, 208)
        TRSWidg.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.gridLayout = QtWidgets.QGridLayout(TRSWidg)
        self.gridLayout.setObjectName("gridLayout")
        self.label_sequencerName = QtWidgets.QLabel(TRSWidg)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_sequencerName.sizePolicy().hasHeightForWidth())
        self.label_sequencerName.setSizePolicy(sizePolicy)
        self.label_sequencerName.setObjectName("label_sequencerName")
        self.gridLayout.addWidget(self.label_sequencerName, 0, 0, 1, 1)
        self.label_8 = QtWidgets.QLabel(TRSWidg)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_8.sizePolicy().hasHeightForWidth())
        self.label_8.setSizePolicy(sizePolicy)
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 1, 0, 1, 1)
        self.label_9 = QtWidgets.QLabel(TRSWidg)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_9.sizePolicy().hasHeightForWidth())
        self.label_9.setSizePolicy(sizePolicy)
        self.label_9.setObjectName("label_9")
        self.gridLayout.addWidget(self.label_9, 1, 1, 1, 1)
        self.label_10 = QtWidgets.QLabel(TRSWidg)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_10.sizePolicy().hasHeightForWidth())
        self.label_10.setSizePolicy(sizePolicy)
        self.label_10.setObjectName("label_10")
        self.gridLayout.addWidget(self.label_10, 1, 2, 1, 1)
        self.label_dwellTime_ms = QtWidgets.QLabel(TRSWidg)
        self.label_dwellTime_ms.setObjectName("label_dwellTime_ms")
        self.gridLayout.addWidget(self.label_dwellTime_ms, 2, 0, 1, 1)
        self.spinBox_nOfBins = QtWidgets.QSpinBox(TRSWidg)
        self.spinBox_nOfBins.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.spinBox_nOfBins.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.spinBox_nOfBins.setKeyboardTracking(False)
        self.spinBox_nOfBins.setMaximum(999999999)
        self.spinBox_nOfBins.setObjectName("spinBox_nOfBins")
        self.gridLayout.addWidget(self.spinBox_nOfBins, 2, 1, 1, 1)
        self.label_nOfBins_set = QtWidgets.QLabel(TRSWidg)
        self.label_nOfBins_set.setObjectName("label_nOfBins_set")
        self.gridLayout.addWidget(self.label_nOfBins_set, 2, 2, 1, 1)
        self.label_3 = QtWidgets.QLabel(TRSWidg)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 3, 0, 1, 1)
        self.spinBox_softBinWidth = QtWidgets.QSpinBox(TRSWidg)
        self.spinBox_softBinWidth.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.spinBox_softBinWidth.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.spinBox_softBinWidth.setKeyboardTracking(False)
        self.spinBox_softBinWidth.setMinimum(10)
        self.spinBox_softBinWidth.setMaximum(100000)
        self.spinBox_softBinWidth.setSingleStep(10)
        self.spinBox_softBinWidth.setObjectName("spinBox_softBinWidth")
        self.gridLayout.addWidget(self.spinBox_softBinWidth, 3, 1, 1, 1)
        self.label_softBinWidth_set = QtWidgets.QLabel(TRSWidg)
        self.label_softBinWidth_set.setObjectName("label_softBinWidth_set")
        self.gridLayout.addWidget(self.label_softBinWidth_set, 3, 2, 1, 1)
        self.label = QtWidgets.QLabel(TRSWidg)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 4, 0, 1, 1)
        self.spinBox_nOfBunches = QtWidgets.QSpinBox(TRSWidg)
        self.spinBox_nOfBunches.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.spinBox_nOfBunches.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.spinBox_nOfBunches.setKeyboardTracking(False)
        self.spinBox_nOfBunches.setMaximum(999999999)
        self.spinBox_nOfBunches.setObjectName("spinBox_nOfBunches")
        self.gridLayout.addWidget(self.spinBox_nOfBunches, 4, 1, 1, 1)
        self.label_nOfBunches_set = QtWidgets.QLabel(TRSWidg)
        self.label_nOfBunches_set.setObjectName("label_nOfBunches_set")
        self.gridLayout.addWidget(self.label_nOfBunches_set, 4, 2, 1, 1)
        self.label_4 = QtWidgets.QLabel(TRSWidg)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 5, 0, 1, 1)
        self.doubleSpinBox_mid_tof = QtWidgets.QDoubleSpinBox(TRSWidg)
        self.doubleSpinBox_mid_tof.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.doubleSpinBox_mid_tof.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.doubleSpinBox_mid_tof.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.doubleSpinBox_mid_tof.setKeyboardTracking(False)
        self.doubleSpinBox_mid_tof.setMaximum(10000000.0)
        self.doubleSpinBox_mid_tof.setObjectName("doubleSpinBox_mid_tof")
        self.gridLayout.addWidget(self.doubleSpinBox_mid_tof, 5, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(TRSWidg)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 6, 0, 1, 1)
        self.doubleSpinBox_gate_width = QtWidgets.QDoubleSpinBox(TRSWidg)
        self.doubleSpinBox_gate_width.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.doubleSpinBox_gate_width.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.doubleSpinBox_gate_width.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.doubleSpinBox_gate_width.setKeyboardTracking(False)
        self.doubleSpinBox_gate_width.setMaximum(10000000.0)
        self.doubleSpinBox_gate_width.setObjectName("doubleSpinBox_gate_width")
        self.gridLayout.addWidget(self.doubleSpinBox_gate_width, 6, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(TRSWidg)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 7, 0, 1, 1)
        self.lineEdit = QtWidgets.QLineEdit(TRSWidg)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout.addWidget(self.lineEdit, 7, 1, 1, 1)
        self.label_softwGates_set = QtWidgets.QLabel(TRSWidg)
        self.label_softwGates_set.setObjectName("label_softwGates_set")
        self.gridLayout.addWidget(self.label_softwGates_set, 7, 2, 1, 1)
        self.label_gate_width_set = QtWidgets.QLabel(TRSWidg)
        self.label_gate_width_set.setObjectName("label_gate_width_set")
        self.gridLayout.addWidget(self.label_gate_width_set, 6, 2, 1, 1)
        self.label_mid_tof_set = QtWidgets.QLabel(TRSWidg)
        self.label_mid_tof_set.setObjectName("label_mid_tof_set")
        self.gridLayout.addWidget(self.label_mid_tof_set, 5, 2, 1, 1)

        self.retranslateUi(TRSWidg)
        QtCore.QMetaObject.connectSlotsByName(TRSWidg)

    def retranslateUi(self, TRSWidg):
        _translate = QtCore.QCoreApplication.translate
        TRSWidg.setWindowTitle(_translate("TRSWidg", "TRSWidg"))
        self.label_sequencerName.setText(_translate("TRSWidg", "time resolved sequencer settings"))
        self.label_8.setText(_translate("TRSWidg", "Parameter"))
        self.label_9.setText(_translate("TRSWidg", "User Input"))
        self.label_10.setText(_translate("TRSWidg", "Set Value"))
        self.label_dwellTime_ms.setText(_translate("TRSWidg", "# 10ns bins"))
        self.label_nOfBins_set.setText(_translate("TRSWidg", "None"))
        self.label_3.setText(_translate("TRSWidg", "software bin width / ns"))
        self.label_softBinWidth_set.setText(_translate("TRSWidg", "None"))
        self.label.setText(_translate("TRSWidg", "# bunches per step"))
        self.label_nOfBunches_set.setText(_translate("TRSWidg", "None"))
        self.label_4.setText(_translate("TRSWidg", "softw. mid TOF / us"))
        self.label_5.setText(_translate("TRSWidg", "softw. gate width / us"))
        self.label_2.setText(_translate("TRSWidg", "software gates scaler delay list / us:"))
        self.lineEdit.setToolTip(_translate("TRSWidg", "<html><head/><body><p>list of delays to the common mid tof for each scaler.</p></body></html>"))
        self.label_softwGates_set.setText(_translate("TRSWidg", "None"))
        self.label_gate_width_set.setText(_translate("TRSWidg", "None"))
        self.label_mid_tof_set.setText(_translate("TRSWidg", "None"))

