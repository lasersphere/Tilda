# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_Simp_Count_Dial.ui'
#
# Created by: PyQt5 UI code generator 5.5
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog_simpleCounterControl(object):
    def setupUi(self, Dialog_simpleCounterControl):
        Dialog_simpleCounterControl.setObjectName("Dialog_simpleCounterControl")
        Dialog_simpleCounterControl.resize(234, 114)
        Dialog_simpleCounterControl.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.verticalLayout = QtWidgets.QVBoxLayout(Dialog_simpleCounterControl)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_5 = QtWidgets.QLabel(Dialog_simpleCounterControl)
        self.label_5.setObjectName("label_5")
        self.verticalLayout.addWidget(self.label_5)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_3 = QtWidgets.QLabel(Dialog_simpleCounterControl)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(Dialog_simpleCounterControl)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 0, 1, 1)
        self.lineEdit_act_pmts = QtWidgets.QLineEdit(Dialog_simpleCounterControl)
        self.lineEdit_act_pmts.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.lineEdit_act_pmts.setObjectName("lineEdit_act_pmts")
        self.gridLayout.addWidget(self.lineEdit_act_pmts, 0, 1, 1, 1)
        self.label_act_pmts_set = QtWidgets.QLabel(Dialog_simpleCounterControl)
        self.label_act_pmts_set.setObjectName("label_act_pmts_set")
        self.gridLayout.addWidget(self.label_act_pmts_set, 0, 2, 1, 1)
        self.label_plotpoints_set = QtWidgets.QLabel(Dialog_simpleCounterControl)
        self.label_plotpoints_set.setObjectName("label_plotpoints_set")
        self.gridLayout.addWidget(self.label_plotpoints_set, 2, 2, 1, 1)
        self.spinBox_plotpoints = QtWidgets.QSpinBox(Dialog_simpleCounterControl)
        self.spinBox_plotpoints.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.spinBox_plotpoints.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.spinBox_plotpoints.setKeyboardTracking(False)
        self.spinBox_plotpoints.setMaximum(10000)
        self.spinBox_plotpoints.setObjectName("spinBox_plotpoints")
        self.gridLayout.addWidget(self.spinBox_plotpoints, 2, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog_simpleCounterControl)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(Dialog_simpleCounterControl)
        self.buttonBox.accepted.connect(Dialog_simpleCounterControl.accept)
        self.buttonBox.rejected.connect(Dialog_simpleCounterControl.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog_simpleCounterControl)

    def retranslateUi(self, Dialog_simpleCounterControl):
        _translate = QtCore.QCoreApplication.translate
        Dialog_simpleCounterControl.setWindowTitle(_translate("Dialog_simpleCounterControl", "SimpleCounterControl"))
        self.label_5.setText(_translate("Dialog_simpleCounterControl", "select active pmt\'s & number of datapoint"))
        self.label_3.setText(_translate("Dialog_simpleCounterControl", "datapoints"))
        self.label_2.setText(_translate("Dialog_simpleCounterControl", "active pmts"))
        self.label_act_pmts_set.setText(_translate("Dialog_simpleCounterControl", "TextLabel"))
        self.label_plotpoints_set.setText(_translate("Dialog_simpleCounterControl", "TextLabel"))

