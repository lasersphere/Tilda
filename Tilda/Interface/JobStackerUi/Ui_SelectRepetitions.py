# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_SelectRepetitions.ui'
#
# Created by: PyQt5 UI code generator 5.5
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog_JobRepetitions(object):
    def setupUi(self, Dialog_JobRepetitions):
        Dialog_JobRepetitions.setObjectName("Dialog_JobRepetitions")
        Dialog_JobRepetitions.resize(258, 69)
        self.verticalLayout = QtWidgets.QVBoxLayout(Dialog_JobRepetitions)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_number_reps = QtWidgets.QLabel(Dialog_JobRepetitions)
        self.label_number_reps.setObjectName("label_number_reps")
        self.horizontalLayout.addWidget(self.label_number_reps)
        self.spinBox_number_reps = QtWidgets.QSpinBox(Dialog_JobRepetitions)
        self.spinBox_number_reps.setObjectName("spinBox_number_reps")
        self.horizontalLayout.addWidget(self.spinBox_number_reps)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog_JobRepetitions)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(Dialog_JobRepetitions)
        self.buttonBox.accepted.connect(Dialog_JobRepetitions.accept)
        self.buttonBox.rejected.connect(Dialog_JobRepetitions.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog_JobRepetitions)

    def retranslateUi(self, Dialog_JobRepetitions):
        _translate = QtCore.QCoreApplication.translate
        Dialog_JobRepetitions.setWindowTitle(_translate("Dialog_JobRepetitions", "Please set job repetitions"))
        self.label_number_reps.setText(_translate("Dialog_JobRepetitions", "Number of repetitions for selected job(s)"))

