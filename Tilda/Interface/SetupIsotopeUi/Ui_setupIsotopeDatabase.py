# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_setupIsotopeDatabase.ui'
#
# Created by: PyQt5 UI code generator 5.5
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_setupIsotopeDatabase(object):
    def setupUi(self, setupIsotopeDatabase):
        setupIsotopeDatabase.setObjectName("setupIsotopeDatabase")
        setupIsotopeDatabase.resize(356, 297)
        self.verticalLayout = QtWidgets.QVBoxLayout(setupIsotopeDatabase)
        self.verticalLayout.setObjectName("verticalLayout")
        self.widget = QtWidgets.QWidget(setupIsotopeDatabase)
        self.widget.setObjectName("widget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.tableWidget_iso_pars = QtWidgets.QTableWidget(self.widget)
        self.tableWidget_iso_pars.setObjectName("tableWidget_iso_pars")
        self.tableWidget_iso_pars.setColumnCount(0)
        self.tableWidget_iso_pars.setRowCount(0)
        self.verticalLayout_2.addWidget(self.tableWidget_iso_pars)
        self.verticalLayout.addWidget(self.widget)
        self.buttonBox = QtWidgets.QDialogButtonBox(setupIsotopeDatabase)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(setupIsotopeDatabase)
        self.buttonBox.accepted.connect(setupIsotopeDatabase.accept)
        self.buttonBox.rejected.connect(setupIsotopeDatabase.reject)
        QtCore.QMetaObject.connectSlotsByName(setupIsotopeDatabase)

    def retranslateUi(self, setupIsotopeDatabase):
        _translate = QtCore.QCoreApplication.translate
        setupIsotopeDatabase.setWindowTitle(_translate("setupIsotopeDatabase", "Dialog"))

