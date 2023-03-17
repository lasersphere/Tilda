# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_ChooseDmmWidget.ui'
#
# Created by: PyQt5 UI code generator 5.5
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(296, 110)
        self.formLayout = QtWidgets.QFormLayout(Form)
        self.formLayout.setObjectName("formLayout")
        self.label = QtWidgets.QLabel(Form)
        self.label.setObjectName("label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.comboBox_choose_dmm = QtWidgets.QComboBox(Form)
        self.comboBox_choose_dmm.setObjectName("comboBox_choose_dmm")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.comboBox_choose_dmm)
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.lineEdit_address_dmm = QtWidgets.QLineEdit(Form)
        self.lineEdit_address_dmm.setObjectName("lineEdit_address_dmm")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lineEdit_address_dmm)
        self.pushButton_initialize = QtWidgets.QPushButton(Form)
        self.pushButton_initialize.setObjectName("pushButton_initialize")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.pushButton_initialize)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "choose dmm type:"))
        self.label_2.setText(_translate("Form", "set address"))
        self.pushButton_initialize.setText(_translate("Form", "initialize"))

