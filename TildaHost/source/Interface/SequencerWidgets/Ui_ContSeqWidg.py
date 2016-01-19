# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_ContSeqWidg.ui'
#
# Created by: PyQt5 UI code generator 5.5
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ContSeqWidg(object):
    def setupUi(self, ContSeqWidg):
        ContSeqWidg.setObjectName("ContSeqWidg")
        ContSeqWidg.resize(302, 78)
        self.verticalLayout = QtWidgets.QVBoxLayout(ContSeqWidg)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_sequencerName = QtWidgets.QLabel(ContSeqWidg)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_sequencerName.sizePolicy().hasHeightForWidth())
        self.label_sequencerName.setSizePolicy(sizePolicy)
        self.label_sequencerName.setObjectName("label_sequencerName")
        self.verticalLayout.addWidget(self.label_sequencerName)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_dwellTime_ms = QtWidgets.QLabel(ContSeqWidg)
        self.label_dwellTime_ms.setObjectName("label_dwellTime_ms")
        self.gridLayout.addWidget(self.label_dwellTime_ms, 1, 0, 1, 1)
        self.doubleSpinBox_dwellTime_ms = QtWidgets.QDoubleSpinBox(ContSeqWidg)
        self.doubleSpinBox_dwellTime_ms.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.doubleSpinBox_dwellTime_ms.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.doubleSpinBox_dwellTime_ms.setKeyboardTracking(False)
        self.doubleSpinBox_dwellTime_ms.setDecimals(5)
        self.doubleSpinBox_dwellTime_ms.setMaximum(1000.0)
        self.doubleSpinBox_dwellTime_ms.setObjectName("doubleSpinBox_dwellTime_ms")
        self.gridLayout.addWidget(self.doubleSpinBox_dwellTime_ms, 1, 1, 1, 1)
        self.label_dwellTime_ms_2 = QtWidgets.QLabel(ContSeqWidg)
        self.label_dwellTime_ms_2.setObjectName("label_dwellTime_ms_2")
        self.gridLayout.addWidget(self.label_dwellTime_ms_2, 1, 2, 1, 1)
        self.label_8 = QtWidgets.QLabel(ContSeqWidg)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_8.sizePolicy().hasHeightForWidth())
        self.label_8.setSizePolicy(sizePolicy)
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 0, 0, 1, 1)
        self.label_9 = QtWidgets.QLabel(ContSeqWidg)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_9.sizePolicy().hasHeightForWidth())
        self.label_9.setSizePolicy(sizePolicy)
        self.label_9.setObjectName("label_9")
        self.gridLayout.addWidget(self.label_9, 0, 1, 1, 1)
        self.label_10 = QtWidgets.QLabel(ContSeqWidg)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_10.sizePolicy().hasHeightForWidth())
        self.label_10.setSizePolicy(sizePolicy)
        self.label_10.setObjectName("label_10")
        self.gridLayout.addWidget(self.label_10, 0, 2, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)

        self.retranslateUi(ContSeqWidg)
        QtCore.QMetaObject.connectSlotsByName(ContSeqWidg)

    def retranslateUi(self, ContSeqWidg):
        _translate = QtCore.QCoreApplication.translate
        ContSeqWidg.setWindowTitle(_translate("ContSeqWidg", "ContSeqWidg"))
        self.label_sequencerName.setText(_translate("ContSeqWidg", "Continous Sequencer Settings"))
        self.label_dwellTime_ms.setText(_translate("ContSeqWidg", "dwell time [ms]"))
        self.label_dwellTime_ms_2.setText(_translate("ContSeqWidg", "None"))
        self.label_8.setText(_translate("ContSeqWidg", "Parameter"))
        self.label_9.setText(_translate("ContSeqWidg", "User Input"))
        self.label_10.setText(_translate("ContSeqWidg", "Set Value"))

