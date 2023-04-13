# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_Stop.ui'
#
# Created by: PyQt5 UI code generator 5.5
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_StopUi(object):
    def setupUi(self, StopUi):
        StopUi.setObjectName("StopUi")
        StopUi.resize(379, 76)
        self.formLayout = QtWidgets.QFormLayout(StopUi)
        self.formLayout.setObjectName("formLayout")
        self.label = QtWidgets.QLabel(StopUi)
        self.label.setObjectName("label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.lineEdit = QtWidgets.QLineEdit(StopUi)
        self.lineEdit.setObjectName("lineEdit")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit)
        self.buttonBox = QtWidgets.QDialogButtonBox(StopUi)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.SpanningRole, self.buttonBox)

        self.retranslateUi(StopUi)
        self.buttonBox.accepted.connect(StopUi.accept)
        self.buttonBox.rejected.connect(StopUi.reject)
        QtCore.QMetaObject.connectSlotsByName(StopUi)

    def retranslateUi(self, StopUi):
        _translate = QtCore.QCoreApplication.translate
        StopUi.setWindowTitle(_translate("StopUi", "Dialog"))
        self.label.setText(_translate("StopUi", "active Channels when stopped:"))

