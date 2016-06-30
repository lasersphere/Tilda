# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_Averager.ui'
#
# Created by: PyQt5 UI code generator 5.5
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Averager(object):
    def setupUi(self, Averager):
        Averager.setObjectName("Averager")
        Averager.resize(616, 489)
        self.verticalLayout = QtWidgets.QVBoxLayout(Averager)
        self.verticalLayout.setObjectName("verticalLayout")
        self.runSelect = QtWidgets.QComboBox(Averager)
        self.runSelect.setObjectName("runSelect")
        self.verticalLayout.addWidget(self.runSelect)
        self.isoSelect = QtWidgets.QComboBox(Averager)
        self.isoSelect.setObjectName("isoSelect")
        self.verticalLayout.addWidget(self.isoSelect)
        self.parameter = QtWidgets.QComboBox(Averager)
        self.parameter.setObjectName("parameter")
        self.verticalLayout.addWidget(self.parameter)
        self.fileList = QtWidgets.QListWidget(Averager)
        self.fileList.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.fileList.setObjectName("fileList")
        self.verticalLayout.addWidget(self.fileList)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.label_4 = QtWidgets.QLabel(Averager)
        self.label_4.setWordWrap(True)
        self.label_4.setObjectName("label_4")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.result = QtWidgets.QLineEdit(Averager)
        self.result.setReadOnly(True)
        self.result.setObjectName("result")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.result)
        self.label = QtWidgets.QLabel(Averager)
        self.label.setWordWrap(True)
        self.label.setObjectName("label")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label)
        self.rChi = QtWidgets.QLineEdit(Averager)
        self.rChi.setReadOnly(True)
        self.rChi.setObjectName("rChi")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.rChi)
        self.label_2 = QtWidgets.QLabel(Averager)
        self.label_2.setWordWrap(True)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.statErr = QtWidgets.QLineEdit(Averager)
        self.statErr.setReadOnly(True)
        self.statErr.setObjectName("statErr")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.statErr)
        self.label_3 = QtWidgets.QLabel(Averager)
        self.label_3.setWordWrap(True)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.systErr = QtWidgets.QLineEdit(Averager)
        self.systErr.setReadOnly(True)
        self.systErr.setObjectName("systErr")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.systErr)
        self.verticalLayout.addLayout(self.formLayout)
        self.bsave = QtWidgets.QPushButton(Averager)
        self.bsave.setObjectName("bsave")
        self.verticalLayout.addWidget(self.bsave)

        self.retranslateUi(Averager)
        QtCore.QMetaObject.connectSlotsByName(Averager)

    def retranslateUi(self, Averager):
        _translate = QtCore.QCoreApplication.translate
        Averager.setWindowTitle(_translate("Averager", "Form"))
        self.label_4.setText(_translate("Averager", "result"))
        self.label.setText(_translate("Averager", "reduced Chi^2"))
        self.label_2.setText(_translate("Averager", "statistic error"))
        self.label_3.setText(_translate("Averager", "systematic error"))
        self.bsave.setText(_translate("Averager", "Save and Plot"))

