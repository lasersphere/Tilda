# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_Alive.ui'
#
# Created by: PyQt5 UI code generator 5.5
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Alive(object):
    def setupUi(self, Alive):
        Alive.setObjectName("Alive")
        Alive.resize(616, 489)
        self.verticalLayout = QtWidgets.QVBoxLayout(Alive)
        self.verticalLayout.setObjectName("verticalLayout")
        self.formLayout_2 = QtWidgets.QFormLayout()
        self.formLayout_2.setObjectName("formLayout_2")
        self.label_7 = QtWidgets.QLabel(Alive)
        self.label_7.setObjectName("label_7")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_7)
        self.runSelect = QtWidgets.QComboBox(Alive)
        self.runSelect.setObjectName("runSelect")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.runSelect)
        self.label_5 = QtWidgets.QLabel(Alive)
        self.label_5.setObjectName("label_5")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_5)
        self.isoSelect = QtWidgets.QComboBox(Alive)
        self.isoSelect.setObjectName("isoSelect")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.isoSelect)
        self.label_6 = QtWidgets.QLabel(Alive)
        self.label_6.setObjectName("label_6")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.isoSelect_2 = QtWidgets.QComboBox(Alive)
        self.isoSelect_2.setObjectName("isoSelect_2")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.isoSelect_2)
        self.verticalLayout.addLayout(self.formLayout_2)
        self.pB_compareAuto = QtWidgets.QPushButton(Alive)
        self.pB_compareAuto.setObjectName("pB_compareAuto")
        self.verticalLayout.addWidget(self.pB_compareAuto)
        self.label_8 = QtWidgets.QLabel(Alive)
        self.label_8.setObjectName("label_8")
        self.verticalLayout.addWidget(self.label_8)
        self.fileList = QtWidgets.QListWidget(Alive)
        self.fileList.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.fileList.setObjectName("fileList")
        self.verticalLayout.addWidget(self.fileList)
        self.label_9 = QtWidgets.QLabel(Alive)
        self.label_9.setObjectName("label_9")
        self.verticalLayout.addWidget(self.label_9)
        self.fileList_2 = QtWidgets.QListWidget(Alive)
        self.fileList_2.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.fileList_2.setObjectName("fileList_2")
        self.verticalLayout.addWidget(self.fileList_2)
        self.pB_compareIndividual = QtWidgets.QPushButton(Alive)
        self.pB_compareIndividual.setObjectName("pB_compareIndividual")
        self.verticalLayout.addWidget(self.pB_compareIndividual)

        self.retranslateUi(Alive)
        QtCore.QMetaObject.connectSlotsByName(Alive)

    def retranslateUi(self, Alive):
        _translate = QtCore.QCoreApplication.translate
        Alive.setWindowTitle(_translate("Alive", "Form"))
        self.label_7.setText(_translate("Alive", "Select Run"))
        self.label_5.setText(_translate("Alive", "Define Reference Measurement"))
        self.label_6.setText(_translate("Alive", "Define HV Measurement"))
        self.pB_compareAuto.setText(_translate("Alive", "compare with next/previous reference"))
        self.label_8.setText(_translate("Alive", "select reference measurements"))
        self.label_9.setText(_translate("Alive", "select HV measurements"))
        self.pB_compareIndividual.setText(_translate("Alive", "compare individually"))

