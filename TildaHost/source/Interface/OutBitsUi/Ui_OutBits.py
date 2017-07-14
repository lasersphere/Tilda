# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_OutBits.ui'
#
# Created by: PyQt5 UI code generator 5.5
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_outbits(object):
    def setupUi(self, outbits):
        outbits.setObjectName("outbits")
        outbits.resize(454, 233)
        outbits.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.centralwidget = QtWidgets.QWidget(outbits)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.listWidget_outbits = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget_outbits.setObjectName("listWidget_outbits")
        self.horizontalLayout.addWidget(self.listWidget_outbits)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.pushButton_add_outbit = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_add_outbit.setObjectName("pushButton_add_outbit")
        self.verticalLayout_2.addWidget(self.pushButton_add_outbit)
        self.pushButton_edit_selected = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_edit_selected.setObjectName("pushButton_edit_selected")
        self.verticalLayout_2.addWidget(self.pushButton_edit_selected)
        self.pushButton_remove_selected = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_remove_selected.setObjectName("pushButton_remove_selected")
        self.verticalLayout_2.addWidget(self.pushButton_remove_selected)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem)
        self.pushButton_ok = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_ok.setObjectName("pushButton_ok")
        self.verticalLayout_2.addWidget(self.pushButton_ok)
        self.pushButton_cancel = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_cancel.setObjectName("pushButton_cancel")
        self.verticalLayout_2.addWidget(self.pushButton_cancel)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        outbits.setCentralWidget(self.centralwidget)

        self.retranslateUi(outbits)
        QtCore.QMetaObject.connectSlotsByName(outbits)

    def retranslateUi(self, outbits):
        _translate = QtCore.QCoreApplication.translate
        outbits.setWindowTitle(_translate("outbits", "outbits"))
        self.pushButton_add_outbit.setText(_translate("outbits", "add Outbit"))
        self.pushButton_edit_selected.setText(_translate("outbits", "edit selected"))
        self.pushButton_remove_selected.setText(_translate("outbits", "remove selected"))
        self.pushButton_ok.setText(_translate("outbits", "ok"))
        self.pushButton_cancel.setText(_translate("outbits", "cancel"))

