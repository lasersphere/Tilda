# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'E:\Users\Patrick\Documents\Python Projects\Tilda\Tilda\Interface\ArithmeticsUi\Ui_AddObservable.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_AddObservable(object):
    def setupUi(self, AddObservable):
        AddObservable.setObjectName("AddObservable")
        AddObservable.resize(375, 96)
        self.verticalLayout = QtWidgets.QVBoxLayout(AddObservable)
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(AddObservable)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 1, 1, 1)
        self.le_value = QtWidgets.QLineEdit(AddObservable)
        self.le_value.setObjectName("le_value")
        self.gridLayout.addWidget(self.le_value, 1, 2, 1, 1)
        self.label_2 = QtWidgets.QLabel(AddObservable)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 2, 1, 1)
        self.le_name = QtWidgets.QLineEdit(AddObservable)
        self.le_name.setObjectName("le_name")
        self.gridLayout.addWidget(self.le_name, 1, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(AddObservable)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 0, 3, 1, 1)
        self.comboBox = QtWidgets.QComboBox(AddObservable)
        self.comboBox.setMinimumSize(QtCore.QSize(100, 0))
        self.comboBox.setObjectName("comboBox")
        self.gridLayout.addWidget(self.comboBox, 1, 3, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.pb_add = QtWidgets.QPushButton(AddObservable)
        self.pb_add.setObjectName("pb_add")
        self.horizontalLayout.addWidget(self.pb_add)
        self.pb_cancel = QtWidgets.QPushButton(AddObservable)
        self.pb_cancel.setObjectName("pb_cancel")
        self.horizontalLayout.addWidget(self.pb_cancel)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(AddObservable)
        QtCore.QMetaObject.connectSlotsByName(AddObservable)

    def retranslateUi(self, AddObservable):
        _translate = QtCore.QCoreApplication.translate
        AddObservable.setWindowTitle(_translate("AddObservable", "Add Frequency"))
        self.label.setText(_translate("AddObservable", "Name"))
        self.label_2.setText(_translate("AddObservable", "Value"))
        self.label_3.setText(_translate("AddObservable", "Source"))
        self.pb_add.setText(_translate("AddObservable", "OK"))
        self.pb_cancel.setText(_translate("AddObservable", "Cancel"))
