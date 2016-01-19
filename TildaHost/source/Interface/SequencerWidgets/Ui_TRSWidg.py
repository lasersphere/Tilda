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
        TRSWidg.resize(346, 119)
        self.verticalLayout = QtWidgets.QVBoxLayout(TRSWidg)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_sequencerName = QtWidgets.QLabel(TRSWidg)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_sequencerName.sizePolicy().hasHeightForWidth())
        self.label_sequencerName.setSizePolicy(sizePolicy)
        self.label_sequencerName.setObjectName("label_sequencerName")
        self.verticalLayout.addWidget(self.label_sequencerName)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_dwellTime_ms = QtWidgets.QLabel(TRSWidg)
        self.label_dwellTime_ms.setObjectName("label_dwellTime_ms")
        self.gridLayout.addWidget(self.label_dwellTime_ms, 1, 0, 1, 1)
        self.label_9 = QtWidgets.QLabel(TRSWidg)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_9.sizePolicy().hasHeightForWidth())
        self.label_9.setSizePolicy(sizePolicy)
        self.label_9.setObjectName("label_9")
        self.gridLayout.addWidget(self.label_9, 0, 1, 1, 1)
        self.label_8 = QtWidgets.QLabel(TRSWidg)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_8.sizePolicy().hasHeightForWidth())
        self.label_8.setSizePolicy(sizePolicy)
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 0, 0, 1, 1)
        self.label_nOfBins_set = QtWidgets.QLabel(TRSWidg)
        self.label_nOfBins_set.setObjectName("label_nOfBins_set")
        self.gridLayout.addWidget(self.label_nOfBins_set, 1, 2, 1, 1)
        self.label_10 = QtWidgets.QLabel(TRSWidg)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_10.sizePolicy().hasHeightForWidth())
        self.label_10.setSizePolicy(sizePolicy)
        self.label_10.setObjectName("label_10")
        self.gridLayout.addWidget(self.label_10, 0, 2, 1, 1)
        self.spinBox_nOfBins = QtWidgets.QSpinBox(TRSWidg)
        self.spinBox_nOfBins.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.spinBox_nOfBins.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.spinBox_nOfBins.setKeyboardTracking(False)
        self.spinBox_nOfBins.setMaximum(999999999)
        self.spinBox_nOfBins.setObjectName("spinBox_nOfBins")
        self.gridLayout.addWidget(self.spinBox_nOfBins, 1, 1, 1, 1)
        self.label = QtWidgets.QLabel(TRSWidg)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 2, 0, 1, 1)
        self.spinBox_nOfBunches = QtWidgets.QSpinBox(TRSWidg)
        self.spinBox_nOfBunches.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.spinBox_nOfBunches.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.spinBox_nOfBunches.setKeyboardTracking(False)
        self.spinBox_nOfBunches.setMaximum(999999999)
        self.spinBox_nOfBunches.setObjectName("spinBox_nOfBunches")
        self.gridLayout.addWidget(self.spinBox_nOfBunches, 2, 1, 1, 1)
        self.label_nOfBunches_set = QtWidgets.QLabel(TRSWidg)
        self.label_nOfBunches_set.setObjectName("label_nOfBunches_set")
        self.gridLayout.addWidget(self.label_nOfBunches_set, 2, 2, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)

        self.retranslateUi(TRSWidg)
        QtCore.QMetaObject.connectSlotsByName(TRSWidg)

    def retranslateUi(self, TRSWidg):
        _translate = QtCore.QCoreApplication.translate
        TRSWidg.setWindowTitle(_translate("TRSWidg", "TRSWidg"))
        self.label_sequencerName.setText(_translate("TRSWidg", "time resolved sequencer settings"))
        self.label_dwellTime_ms.setText(_translate("TRSWidg", "# bins"))
        self.label_9.setText(_translate("TRSWidg", "User Input"))
        self.label_8.setText(_translate("TRSWidg", "Parameter"))
        self.label_nOfBins_set.setText(_translate("TRSWidg", "None"))
        self.label_10.setText(_translate("TRSWidg", "Set Value"))
        self.label.setText(_translate("TRSWidg", "# bunches per step"))
        self.label_nOfBunches_set.setText(_translate("TRSWidg", "None"))

