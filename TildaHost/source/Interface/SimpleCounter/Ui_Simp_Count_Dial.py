# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_Simp_Count_Dial.ui'
#
# Created by: PyQt5 UI code generator 5.5
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog_simpleCounterControl(object):
    def setupUi(self, Dialog_simpleCounterControl):
        Dialog_simpleCounterControl.setObjectName("Dialog_simpleCounterControl")
        Dialog_simpleCounterControl.resize(256, 64)
        self.verticalLayout = QtWidgets.QVBoxLayout(Dialog_simpleCounterControl)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(Dialog_simpleCounterControl)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog_simpleCounterControl)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(Dialog_simpleCounterControl)
        self.buttonBox.accepted.connect(Dialog_simpleCounterControl.accept)
        self.buttonBox.rejected.connect(Dialog_simpleCounterControl.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog_simpleCounterControl)

    def retranslateUi(self, Dialog_simpleCounterControl):
        _translate = QtCore.QCoreApplication.translate
        Dialog_simpleCounterControl.setWindowTitle(_translate("Dialog_simpleCounterControl", "SimpleCounterControl"))
        self.label.setText(_translate("Dialog_simpleCounterControl", "Press ok/cancel to stop the Simple Counter"))

