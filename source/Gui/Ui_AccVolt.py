# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_AccVolt.ui'
#
# Created by: PyQt5 UI code generator 5.5
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_AccVolt(object):
    def setupUi(self, AccVolt):
        AccVolt.setObjectName("AccVolt")
        AccVolt.resize(400, 300)
        self.verticalLayout = QtWidgets.QVBoxLayout(AccVolt)
        self.verticalLayout.setObjectName("verticalLayout")
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.label = QtWidgets.QLabel(AccVolt)
        self.label.setObjectName("label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.lineEdit_avg = QtWidgets.QLineEdit(AccVolt)
        self.lineEdit_avg.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.lineEdit_avg.setObjectName("lineEdit_avg")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit_avg)
        self.label_2 = QtWidgets.QLabel(AccVolt)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.lineEdit_rChi2 = QtWidgets.QLineEdit(AccVolt)
        self.lineEdit_rChi2.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.lineEdit_rChi2.setObjectName("lineEdit_rChi2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lineEdit_rChi2)
        self.label_3 = QtWidgets.QLabel(AccVolt)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.lineEdit_statErr = QtWidgets.QLineEdit(AccVolt)
        self.lineEdit_statErr.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.lineEdit_statErr.setObjectName("lineEdit_statErr")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.lineEdit_statErr)
        self.label_4 = QtWidgets.QLabel(AccVolt)
        self.label_4.setObjectName("label_4")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.lineEdit_readErr = QtWidgets.QLineEdit(AccVolt)
        self.lineEdit_readErr.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.lineEdit_readErr.setObjectName("lineEdit_readErr")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.lineEdit_readErr)
        self.label_5 = QtWidgets.QLabel(AccVolt)
        self.label_5.setObjectName("label_5")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_5)
        self.lineEdit_savedto = QtWidgets.QLineEdit(AccVolt)
        self.lineEdit_savedto.setObjectName("lineEdit_savedto")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.lineEdit_savedto)
        self.dateTimeEdit_start = QtWidgets.QDateTimeEdit(AccVolt)
        self.dateTimeEdit_start.setObjectName("dateTimeEdit_start")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.dateTimeEdit_start)
        self.label_6 = QtWidgets.QLabel(AccVolt)
        self.label_6.setObjectName("label_6")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.label_7 = QtWidgets.QLabel(AccVolt)
        self.label_7.setObjectName("label_7")
        self.formLayout.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_7)
        self.dateTimeEdit_end = QtWidgets.QDateTimeEdit(AccVolt)
        self.dateTimeEdit_end.setObjectName("dateTimeEdit_end")
        self.formLayout.setWidget(6, QtWidgets.QFormLayout.FieldRole, self.dateTimeEdit_end)
        self.verticalLayout.addLayout(self.formLayout)
        self.pushButton = QtWidgets.QPushButton(AccVolt)
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout.addWidget(self.pushButton)

        self.retranslateUi(AccVolt)
        QtCore.QMetaObject.connectSlotsByName(AccVolt)

    def retranslateUi(self, AccVolt):
        _translate = QtCore.QCoreApplication.translate
        AccVolt.setWindowTitle(_translate("AccVolt", "Form"))
        self.label.setText(_translate("AccVolt", "average"))
        self.label_2.setText(_translate("AccVolt", "reduced Chi^2"))
        self.label_3.setText(_translate("AccVolt", "statistic error"))
        self.label_4.setText(_translate("AccVolt", "uncertainty of reading"))
        self.label_5.setText(_translate("AccVolt", "saved to:"))
        self.label_6.setText(_translate("AccVolt", "from:"))
        self.label_7.setText(_translate("AccVolt", "until:"))
        self.pushButton.setText(_translate("AccVolt", "PlotAccVolt"))

