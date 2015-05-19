# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_InteractiveFit.ui'
#
# Created: Wed Jun 11 15:26:34 2014
#      by: PyQt5 UI code generator 5.2.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_InteractiveFit(object):
    def setupUi(self, InteractiveFit):
        InteractiveFit.setObjectName("InteractiveFit")
        InteractiveFit.resize(716, 475)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(InteractiveFit)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.splitter = QtWidgets.QSplitter(InteractiveFit)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setObjectName("splitter")
        self.layoutWidget = QtWidgets.QWidget(self.splitter)
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.runSelect = QtWidgets.QComboBox(self.layoutWidget)
        self.runSelect.setObjectName("runSelect")
        self.verticalLayout.addWidget(self.runSelect)
        self.isoFilter = QtWidgets.QComboBox(self.layoutWidget)
        self.isoFilter.setObjectName("isoFilter")
        self.verticalLayout.addWidget(self.isoFilter)
        self.fileList = QtWidgets.QListWidget(self.layoutWidget)
        self.fileList.setObjectName("fileList")
        self.verticalLayout.addWidget(self.fileList)
        self.bLoad = QtWidgets.QPushButton(self.layoutWidget)
        self.bLoad.setObjectName("bLoad")
        self.verticalLayout.addWidget(self.bLoad)
        self.layoutWidget1 = QtWidgets.QWidget(self.splitter)
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.layoutWidget1)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.parTable = QtWidgets.QTableWidget(self.layoutWidget1)
        self.parTable.setColumnCount(3)
        self.parTable.setObjectName("parTable")
        self.parTable.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.parTable.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.parTable.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.parTable.setHorizontalHeaderItem(2, item)
        self.verticalLayout_2.addWidget(self.parTable)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.bFit = QtWidgets.QPushButton(self.layoutWidget1)
        self.bFit.setObjectName("bFit")
        self.horizontalLayout.addWidget(self.bFit)
        self.bReset = QtWidgets.QPushButton(self.layoutWidget1)
        self.bReset.setObjectName("bReset")
        self.horizontalLayout.addWidget(self.bReset)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.horizontalLayout_2.addWidget(self.splitter)

        self.retranslateUi(InteractiveFit)
        QtCore.QMetaObject.connectSlotsByName(InteractiveFit)

    def retranslateUi(self, InteractiveFit):
        _translate = QtCore.QCoreApplication.translate
        InteractiveFit.setWindowTitle(_translate("InteractiveFit", "Form"))
        self.bLoad.setText(_translate("InteractiveFit", "Load"))
        item = self.parTable.horizontalHeaderItem(0)
        item.setText(_translate("InteractiveFit", "Parameter"))
        item = self.parTable.horizontalHeaderItem(1)
        item.setText(_translate("InteractiveFit", "Value"))
        item = self.parTable.horizontalHeaderItem(2)
        item.setText(_translate("InteractiveFit", "Fixed"))
        self.bFit.setText(_translate("InteractiveFit", "Fit"))
        self.bReset.setText(_translate("InteractiveFit", "Reset"))

