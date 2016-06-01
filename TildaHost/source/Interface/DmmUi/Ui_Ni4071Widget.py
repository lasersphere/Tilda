# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_Ni4071Widget.ui'
#
# Created by: PyQt5 UI code generator 5.5
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_form_layout(object):
    def setupUi(self, form_layout):
        form_layout.setObjectName("form_layout")
        form_layout.resize(325, 218)
        self.formLayout = QtWidgets.QFormLayout(form_layout)
        self.formLayout.setObjectName("formLayout")
        self.label = QtWidgets.QLabel(form_layout)
        self.label.setObjectName("label")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label)
        self.lcdNumber = QtWidgets.QLCDNumber(form_layout)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lcdNumber.sizePolicy().hasHeightForWidth())
        self.lcdNumber.setSizePolicy(sizePolicy)
        self.lcdNumber.setDigitCount(8)
        self.lcdNumber.setObjectName("lcdNumber")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.lcdNumber)
        self.spinBox = QtWidgets.QSpinBox(form_layout)
        self.spinBox.setObjectName("spinBox")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.spinBox)
        self.label_2 = QtWidgets.QLabel(form_layout)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_2)

        self.retranslateUi(form_layout)
        QtCore.QMetaObject.connectSlotsByName(form_layout)

    def retranslateUi(self, form_layout):
        _translate = QtCore.QCoreApplication.translate
        form_layout.setWindowTitle(_translate("form_layout", "Form"))
        self.label.setText(_translate("form_layout", "TextLabel"))
        self.label_2.setText(_translate("form_layout", "last reading [V]:"))

