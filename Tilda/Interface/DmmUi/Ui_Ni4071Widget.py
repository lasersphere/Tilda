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
        form_layout.resize(279, 218)
        self.verticalLayout = QtWidgets.QVBoxLayout(form_layout)
        self.verticalLayout.setObjectName("verticalLayout")
        self.formLayout_reading_and_buttons = QtWidgets.QFormLayout()
        self.formLayout_reading_and_buttons.setObjectName("formLayout_reading_and_buttons")
        self.label_2 = QtWidgets.QLabel(form_layout)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setMinimumSize(QtCore.QSize(150, 0))
        self.label_2.setObjectName("label_2")
        self.formLayout_reading_and_buttons.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.lcdNumber = QtWidgets.QLCDNumber(form_layout)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lcdNumber.sizePolicy().hasHeightForWidth())
        self.lcdNumber.setSizePolicy(sizePolicy)
        self.lcdNumber.setMinimumSize(QtCore.QSize(0, 50))
        self.lcdNumber.setDigitCount(8)
        self.lcdNumber.setObjectName("lcdNumber")
        self.formLayout_reading_and_buttons.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lcdNumber)
        self.verticalLayout.addLayout(self.formLayout_reading_and_buttons)
        self.formLayout_config_values = QtWidgets.QFormLayout()
        self.formLayout_config_values.setObjectName("formLayout_config_values")
        self.verticalLayout.addLayout(self.formLayout_config_values)

        self.retranslateUi(form_layout)
        QtCore.QMetaObject.connectSlotsByName(form_layout)

    def retranslateUi(self, form_layout):
        _translate = QtCore.QCoreApplication.translate
        form_layout.setWindowTitle(_translate("form_layout", "Form"))
        self.label_2.setText(_translate("form_layout", "last reading [V]:"))

