# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_SetVoltage.ui'
#
# Created by: PyQt5 UI code generator 5.5
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(212, 118)
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.label_lastSetVolt = QtWidgets.QLabel(Dialog)
        self.label_lastSetVolt.setObjectName("label_lastSetVolt")
        self.gridLayout.addWidget(self.label_lastSetVolt, 0, 1, 1, 1)
        self.label_targetVoltage = QtWidgets.QLabel(Dialog)
        self.label_targetVoltage.setObjectName("label_targetVoltage")
        self.gridLayout.addWidget(self.label_targetVoltage, 2, 1, 1, 1)
        self.label_lastVoltageSetAt = QtWidgets.QLabel(Dialog)
        self.label_lastVoltageSetAt.setObjectName("label_lastVoltageSetAt")
        self.gridLayout.addWidget(self.label_lastVoltageSetAt, 1, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 1, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 3, 0, 1, 1)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 2, 0, 1, 1)
        self.label_voltReadBack = QtWidgets.QLabel(Dialog)
        self.label_voltReadBack.setObjectName("label_voltReadBack")
        self.gridLayout.addWidget(self.label_voltReadBack, 3, 1, 1, 1)
        self.pushButton_ok = QtWidgets.QPushButton(Dialog)
        self.pushButton_ok.setObjectName("pushButton_ok")
        self.gridLayout.addWidget(self.pushButton_ok, 4, 0, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "SetVoltage"))
        self.label_lastSetVolt.setText(_translate("Dialog", "0"))
        self.label_targetVoltage.setText(_translate("Dialog", "0"))
        self.label_lastVoltageSetAt.setText(_translate("Dialog", "00:00:00"))
        self.label_3.setText(_translate("Dialog", "last voltage set at:"))
        self.label_2.setText(_translate("Dialog", "curent readback:"))
        self.label.setText(_translate("Dialog", "last set voltage:"))
        self.label_4.setText(_translate("Dialog", "new voltage set:"))
        self.label_voltReadBack.setText(_translate("Dialog", "0"))
        self.pushButton_ok.setText(_translate("Dialog", "ok"))

