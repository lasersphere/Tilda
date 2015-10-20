# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_setupIsotope.ui'
#
# Created by: PyQt5 UI code generator 5.5
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_SetupIsotope(object):
    def setupUi(self, SetupIsotope):
        SetupIsotope.setObjectName("SetupIsotope")
        SetupIsotope.resize(272, 115)
        self.verticalLayout = QtWidgets.QVBoxLayout(SetupIsotope)
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_2 = QtWidgets.QLabel(SetupIsotope)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.label_isotope = QtWidgets.QLabel(SetupIsotope)
        self.label_isotope.setObjectName("label_isotope")
        self.gridLayout.addWidget(self.label_isotope, 0, 0, 1, 1)
        self.comboBox_sequencer = QtWidgets.QComboBox(SetupIsotope)
        self.comboBox_sequencer.setObjectName("comboBox_sequencer")
        self.comboBox_sequencer.addItem("")
        self.comboBox_sequencer.addItem("")
        self.comboBox_sequencer.addItem("")
        self.gridLayout.addWidget(self.comboBox_sequencer, 1, 1, 1, 1)
        self.pushButton = QtWidgets.QPushButton(SetupIsotope)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout.addWidget(self.pushButton, 1, 2, 1, 1)
        self.comboBox = QtWidgets.QComboBox(SetupIsotope)
        self.comboBox.setObjectName("comboBox")
        self.gridLayout.addWidget(self.comboBox, 0, 1, 1, 1)
        self.pushButton_2 = QtWidgets.QPushButton(SetupIsotope)
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout.addWidget(self.pushButton_2, 0, 2, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)

        self.retranslateUi(SetupIsotope)
        QtCore.QMetaObject.connectSlotsByName(SetupIsotope)

    def retranslateUi(self, SetupIsotope):
        _translate = QtCore.QCoreApplication.translate
        SetupIsotope.setWindowTitle(_translate("SetupIsotope", "Setup Isotope"))
        self.label_2.setText(_translate("SetupIsotope", "Sequencer:"))
        self.label_isotope.setText(_translate("SetupIsotope", "Isotope:"))
        self.comboBox_sequencer.setItemText(0, _translate("SetupIsotope", "continous"))
        self.comboBox_sequencer.setItemText(1, _translate("SetupIsotope", "time resolved"))
        self.comboBox_sequencer.setItemText(2, _translate("SetupIsotope", "kepco scan"))
        self.pushButton.setText(_translate("SetupIsotope", "initialize sequencer"))
        self.pushButton_2.setText(_translate("SetupIsotope", "add  new to db"))

