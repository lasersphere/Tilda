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
        Batchfitter.resize(767, 669)
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
        self.pushButton_select_all = QtWidgets.QPushButton(Batchfitter)
        self.pushButton_select_all.setObjectName("pushButton_select_all")
        self.verticalLayout.addWidget(self.pushButton_select_all)
        self.bfit = QtWidgets.QPushButton(Batchfitter)
        self.bfit.setObjectName("bfit")
        self.verticalLayout.addWidget(self.bfit)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.label = QtWidgets.QLabel(Batchfitter)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.savePlotAs = QtWidgets.QComboBox(Batchfitter)
        self.savePlotAs.setObjectName("savePlotAs")
        self.savePlotAs.addItem("")
        self.savePlotAs.addItem("")
        self.savePlotAs.addItem("")
        self.horizontalLayout.addWidget(self.savePlotAs)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(Batchfitter)
        QtCore.QMetaObject.connectSlotsByName(Batchfitter)

    def retranslateUi(self, Batchfitter):
        _translate = QtCore.QCoreApplication.translate
        Batchfitter.setWindowTitle(_translate("Batchfitter", "Form"))
        self.pushButton_select_all.setText(_translate("Batchfitter", "select/deselect all"))
        self.bfit.setText(_translate("Batchfitter", "Fit"))
        self.label.setText(_translate("Batchfitter", "Save Plots as"))
        self.savePlotAs.setItemText(0, _translate("Batchfitter", ".png"))
        self.savePlotAs.setItemText(1, _translate("Batchfitter", ".pdf"))
        self.savePlotAs.setItemText(2, _translate("Batchfitter", ".eps"))

