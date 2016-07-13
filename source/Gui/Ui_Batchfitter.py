# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_Batchfitter.ui'
#
# Created by: PyQt5 UI code generator 5.5
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Batchfitter(object):
    def setupUi(self, Batchfitter):
        Batchfitter.setObjectName("Batchfitter")
        Batchfitter.resize(616, 489)
        self.verticalLayout = QtWidgets.QVBoxLayout(Batchfitter)
        self.verticalLayout.setObjectName("verticalLayout")
        self.runSelect = QtWidgets.QComboBox(Batchfitter)
        self.runSelect.setObjectName("runSelect")
        self.verticalLayout.addWidget(self.runSelect)
        self.isoSelect = QtWidgets.QComboBox(Batchfitter)
        self.isoSelect.setObjectName("isoSelect")
        self.verticalLayout.addWidget(self.isoSelect)
        self.fileList = QtWidgets.QListWidget(Batchfitter)
        self.fileList.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.fileList.setObjectName("fileList")
        self.verticalLayout.addWidget(self.fileList)
        self.bfit = QtWidgets.QPushButton(Batchfitter)
        self.bfit.setObjectName("bfit")
        self.verticalLayout.addWidget(self.bfit)

        self.retranslateUi(Batchfitter)
        QtCore.QMetaObject.connectSlotsByName(Batchfitter)

    def retranslateUi(self, Batchfitter):
        _translate = QtCore.QCoreApplication.translate
        Batchfitter.setWindowTitle(_translate("Batchfitter", "Form"))
        self.bfit.setText(_translate("Batchfitter", "Fit"))

