# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_Trigger.ui'
#
# Created by: PyQt5 UI code generator 5.5
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_TriggerUi(object):
    def setupUi(self, TriggerUi):
        TriggerUi.setObjectName("TriggerUi")
        TriggerUi.resize(312, 119)
        self.formLayout = QtWidgets.QFormLayout(TriggerUi)
        self.formLayout.setObjectName("formLayout")
        self.label = QtWidgets.QLabel(TriggerUi)
        self.label.setObjectName("label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.lineEdit_trig_name = QtWidgets.QLineEdit(TriggerUi)
        self.lineEdit_trig_name.setObjectName("lineEdit_trig_name")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit_trig_name)
        self.label_2 = QtWidgets.QLabel(TriggerUi)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.lineEdit_trig_channels = QtWidgets.QLineEdit(TriggerUi)
        self.lineEdit_trig_channels.setObjectName("lineEdit_trig_channels")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lineEdit_trig_channels)
        self.label_3 = QtWidgets.QLabel(TriggerUi)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.lineEdit_act_ch = QtWidgets.QLineEdit(TriggerUi)
        self.lineEdit_act_ch.setObjectName("lineEdit_act_ch")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.lineEdit_act_ch)
        self.buttonBox = QtWidgets.QDialogButtonBox(TriggerUi)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.SpanningRole, self.buttonBox)

        self.retranslateUi(TriggerUi)
        self.buttonBox.accepted.connect(TriggerUi.accept)
        self.buttonBox.rejected.connect(TriggerUi.reject)
        QtCore.QMetaObject.connectSlotsByName(TriggerUi)

    def retranslateUi(self, TriggerUi):
        _translate = QtCore.QCoreApplication.translate
        TriggerUi.setWindowTitle(_translate("TriggerUi", "Dialog"))
        self.label.setText(_translate("TriggerUi", "name:"))
        self.label_2.setText(_translate("TriggerUi", "trigger channels:"))
        self.label_3.setText(_translate("TriggerUi", "active channels while waiting:"))

