# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_Isoshift.ui'
#
# Created by: PyQt5 UI code generator 5.5
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Isoshift(object):
    def setupUi(self, Isoshift):
        Isoshift.setObjectName("Isoshift")
        Isoshift.resize(616, 489)
        self.verticalLayout = QtWidgets.QVBoxLayout(Isoshift)
        self.verticalLayout.setObjectName("verticalLayout")
        self.runSelect = QtWidgets.QComboBox(Isoshift)
        self.runSelect.setObjectName("runSelect")
        self.verticalLayout.addWidget(self.runSelect)
        self.isoSelect = QtWidgets.QComboBox(Isoshift)
        self.isoSelect.setObjectName("isoSelect")
        self.verticalLayout.addWidget(self.isoSelect)
        self.fileList = QtWidgets.QListWidget(Isoshift)
        self.fileList.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.fileList.setObjectName("fileList")
        self.verticalLayout.addWidget(self.fileList)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.label_4 = QtWidgets.QLabel(Isoshift)
        self.label_4.setWordWrap(True)
        self.label_4.setObjectName("label_4")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.result = QtWidgets.QLineEdit(Isoshift)
        self.result.setReadOnly(True)
        self.result.setObjectName("result")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.result)
        self.label = QtWidgets.QLabel(Isoshift)
        self.label.setWordWrap(True)
        self.label.setObjectName("label")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label)
        self.rChi = QtWidgets.QLineEdit(Isoshift)
        self.rChi.setReadOnly(True)
        self.rChi.setObjectName("rChi")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.rChi)
        self.label_2 = QtWidgets.QLabel(Isoshift)
        self.label_2.setWordWrap(True)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.statErr = QtWidgets.QLineEdit(Isoshift)
        self.statErr.setReadOnly(True)
        self.statErr.setObjectName("statErr")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.statErr)
        self.label_3 = QtWidgets.QLabel(Isoshift)
        self.label_3.setWordWrap(True)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.systErr = QtWidgets.QLineEdit(Isoshift)
        self.systErr.setReadOnly(True)
        self.systErr.setObjectName("systErr")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.systErr)
        self.verticalLayout.addLayout(self.formLayout)
        self.bsave = QtWidgets.QPushButton(Isoshift)
        self.bsave.setObjectName("bsave")
        self.verticalLayout.addWidget(self.bsave)

        self.retranslateUi(Isoshift)
        QtCore.QMetaObject.connectSlotsByName(Isoshift)

    def retranslateUi(self, Isoshift):
        _translate = QtCore.QCoreApplication.translate
        Isoshift.setWindowTitle(_translate("Isoshift", "Form"))
        self.label_4.setText(_translate("Isoshift", "result"))
        self.label.setText(_translate("Isoshift", "reduced Chi^2"))
        self.label_2.setText(_translate("Isoshift", "statistic error"))
        self.label_3.setText(_translate("Isoshift", "systematic error"))
        self.bsave.setText(_translate("Isoshift", "Save and Plot"))

