# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_SingleHitDelay.ui'
#
# Created by: PyQt5 UI code generator 5.5
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_single_hit_delay_widg(object):
    def setupUi(self, single_hit_delay_widg):
        single_hit_delay_widg.setObjectName("single_hit_delay_widg")
        single_hit_delay_widg.resize(385, 64)
        single_hit_delay_widg.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.gridLayout = QtWidgets.QGridLayout(single_hit_delay_widg)
        self.gridLayout.setObjectName("gridLayout")
        self.comboBox_trigInputChan = QtWidgets.QComboBox(single_hit_delay_widg)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox_trigInputChan.sizePolicy().hasHeightForWidth())
        self.comboBox_trigInputChan.setSizePolicy(sizePolicy)
        self.comboBox_trigInputChan.setObjectName("comboBox_trigInputChan")
        self.gridLayout.addWidget(self.comboBox_trigInputChan, 0, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(single_hit_delay_widg)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.label = QtWidgets.QLabel(single_hit_delay_widg)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.label_trigInputChan_set = QtWidgets.QLabel(single_hit_delay_widg)
        self.label_trigInputChan_set.setObjectName("label_trigInputChan_set")
        self.gridLayout.addWidget(self.label_trigInputChan_set, 0, 2, 1, 1)
        self.doubleSpinBox_trigDelay_mus = QtWidgets.QDoubleSpinBox(single_hit_delay_widg)
        self.doubleSpinBox_trigDelay_mus.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.doubleSpinBox_trigDelay_mus.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.doubleSpinBox_trigDelay_mus.setKeyboardTracking(False)
        self.doubleSpinBox_trigDelay_mus.setMaximum(99999999999.0)
        self.doubleSpinBox_trigDelay_mus.setObjectName("doubleSpinBox_trigDelay_mus")
        self.gridLayout.addWidget(self.doubleSpinBox_trigDelay_mus, 1, 1, 1, 1)
        self.label_trigDelay_mus_set = QtWidgets.QLabel(single_hit_delay_widg)
        self.label_trigDelay_mus_set.setObjectName("label_trigDelay_mus_set")
        self.gridLayout.addWidget(self.label_trigDelay_mus_set, 1, 2, 1, 1)

        self.retranslateUi(single_hit_delay_widg)
        QtCore.QMetaObject.connectSlotsByName(single_hit_delay_widg)

    def retranslateUi(self, single_hit_delay_widg):
        _translate = QtCore.QCoreApplication.translate
        single_hit_delay_widg.setWindowTitle(_translate("single_hit_delay_widg", "single_hit_delay_widg"))
        self.label_2.setText(_translate("single_hit_delay_widg", "Delay to trigger [µs]"))
        self.label.setText(_translate("single_hit_delay_widg", "trigger input channel"))
        self.label_trigInputChan_set.setText(_translate("single_hit_delay_widg", "TextLabel"))
        self.label_trigDelay_mus_set.setText(_translate("single_hit_delay_widg", "TextLabel"))

