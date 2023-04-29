# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'E:\Users\Patrick\Documents\Python Projects\Tilda\Tilda\Interface\ArithmeticsUi\Ui_Arithmetics.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Arithmetics(object):
    def setupUi(self, Arithmetics):
        Arithmetics.setObjectName("Arithmetics")
        Arithmetics.resize(400, 300)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("E:\\Users\\Patrick\\Documents\\Python Projects\\Tilda\\Tilda\\Interface\\ArithmeticsUi\\../icons/Tilda256.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Arithmetics.setWindowIcon(icon)
        Arithmetics.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.Europe))
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(Arithmetics)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label = QtWidgets.QLabel(Arithmetics)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.le_arith = QtWidgets.QLineEdit(Arithmetics)
        self.le_arith.setObjectName("le_arith")
        self.verticalLayout_2.addWidget(self.le_arith)
        self.label_2 = QtWidgets.QLabel(Arithmetics)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_2.addWidget(self.label_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.list_observables = QtWidgets.QListWidget(Arithmetics)
        self.list_observables.setObjectName("list_observables")
        self.horizontalLayout.addWidget(self.list_observables)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.pb_add = QtWidgets.QPushButton(Arithmetics)
        self.pb_add.setObjectName("pb_add")
        self.verticalLayout.addWidget(self.pb_add)
        self.pb_remove = QtWidgets.QPushButton(Arithmetics)
        self.pb_remove.setObjectName("pb_remove")
        self.verticalLayout.addWidget(self.pb_remove)
        self.pb_edit = QtWidgets.QPushButton(Arithmetics)
        self.pb_edit.setObjectName("pb_edit")
        self.verticalLayout.addWidget(self.pb_edit)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.buttonBox = QtWidgets.QDialogButtonBox(Arithmetics)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout_2.addWidget(self.buttonBox)

        self.retranslateUi(Arithmetics)
        self.buttonBox.accepted.connect(Arithmetics.accept)
        self.buttonBox.rejected.connect(Arithmetics.reject)
        QtCore.QMetaObject.connectSlotsByName(Arithmetics)

    def retranslateUi(self, Arithmetics):
        _translate = QtCore.QCoreApplication.translate
        Arithmetics.setWindowTitle(_translate("Arithmetics", "Frequency"))
        self.label.setText(_translate("Arithmetics", "Observable Arithmetic"))
        self.label_2.setText(_translate("Arithmetics", "Given observable names must correspond to the name used in the arithmetic."))
        self.pb_add.setText(_translate("Arithmetics", "Add Observable"))
        self.pb_remove.setText(_translate("Arithmetics", "Remove"))
        self.pb_edit.setText(_translate("Arithmetics", "Edit"))
