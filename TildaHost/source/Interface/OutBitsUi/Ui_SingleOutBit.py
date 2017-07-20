# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_SingleOutBit.ui'
#
# Created by: PyQt5 UI code generator 5.5
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_EditOutbit(object):
    def setupUi(self, EditOutbit):
        EditOutbit.setObjectName("EditOutbit")
        EditOutbit.resize(343, 153)
        EditOutbit.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.verticalLayout = QtWidgets.QVBoxLayout(EditOutbit)
        self.verticalLayout.setObjectName("verticalLayout")
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.comboBox_bit_sel = QtWidgets.QComboBox(EditOutbit)
        self.comboBox_bit_sel.setObjectName("comboBox_bit_sel")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.comboBox_bit_sel)
        self.label = QtWidgets.QLabel(EditOutbit)
        self.label.setObjectName("label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.comboBox_toggle_on_off = QtWidgets.QComboBox(EditOutbit)
        self.comboBox_toggle_on_off.setObjectName("comboBox_toggle_on_off")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.comboBox_toggle_on_off)
        self.label_3 = QtWidgets.QLabel(EditOutbit)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.label_2 = QtWidgets.QLabel(EditOutbit)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.comboBox_scan_step = QtWidgets.QComboBox(EditOutbit)
        self.comboBox_scan_step.setObjectName("comboBox_scan_step")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.comboBox_scan_step)
        self.spinBox_scan_step_num = QtWidgets.QSpinBox(EditOutbit)
        self.spinBox_scan_step_num.setObjectName("spinBox_scan_step_num")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.spinBox_scan_step_num)
        self.label_4 = QtWidgets.QLabel(EditOutbit)
        self.label_4.setObjectName("label_4")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.verticalLayout.addLayout(self.formLayout)
        self.buttonBox = QtWidgets.QDialogButtonBox(EditOutbit)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(EditOutbit)
        self.buttonBox.accepted.connect(EditOutbit.accept)
        self.buttonBox.rejected.connect(EditOutbit.reject)
        QtCore.QMetaObject.connectSlotsByName(EditOutbit)

    def retranslateUi(self, EditOutbit):
        _translate = QtCore.QCoreApplication.translate
        EditOutbit.setWindowTitle(_translate("EditOutbit", "outbit single cmd"))
        self.label.setText(_translate("EditOutbit", "select a bit:"))
        self.label_3.setText(_translate("EditOutbit", "toggle/on/off:"))
        self.label_2.setText(_translate("EditOutbit", "scan/step:"))
        self.label_4.setText(_translate("EditOutbit", "change in scan/step:"))

