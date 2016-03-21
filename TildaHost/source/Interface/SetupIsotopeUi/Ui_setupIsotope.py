# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_setupIsotope.ui'
#
# Created by: PyQt5 UI code generator 5.5
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_SetupIsotope(object):
    def setupUi(self, SetupIsotope):
        SetupIsotope.setObjectName("SetupIsotope")
        SetupIsotope.resize(381, 146)
        SetupIsotope.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.verticalLayout = QtWidgets.QVBoxLayout(SetupIsotope)
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.pushButton_add_new_to_db = QtWidgets.QPushButton(SetupIsotope)
        self.pushButton_add_new_to_db.setObjectName("pushButton_add_new_to_db")
        self.gridLayout.addWidget(self.pushButton_add_new_to_db, 1, 1, 1, 1)
        self.comboBox_isotope = QtWidgets.QComboBox(SetupIsotope)
        self.comboBox_isotope.setObjectName("comboBox_isotope")
        self.gridLayout.addWidget(self.comboBox_isotope, 0, 1, 1, 1)
        self.pushButton_init_sequencer = QtWidgets.QPushButton(SetupIsotope)
        self.pushButton_init_sequencer.setObjectName("pushButton_init_sequencer")
        self.gridLayout.addWidget(self.pushButton_init_sequencer, 2, 2, 1, 1)
        self.comboBox_sequencer_select = QtWidgets.QComboBox(SetupIsotope)
        self.comboBox_sequencer_select.setObjectName("comboBox_sequencer_select")
        self.gridLayout.addWidget(self.comboBox_sequencer_select, 2, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(SetupIsotope)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 2, 0, 1, 1)
        self.label_isotope = QtWidgets.QLabel(SetupIsotope)
        self.label_isotope.setObjectName("label_isotope")
        self.gridLayout.addWidget(self.label_isotope, 0, 0, 1, 1)
        self.label_sequencer_state = QtWidgets.QLabel(SetupIsotope)
        self.label_sequencer_state.setObjectName("label_sequencer_state")
        self.gridLayout.addWidget(self.label_sequencer_state, 3, 2, 1, 1)
        self.label = QtWidgets.QLabel(SetupIsotope)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 3, 0, 1, 1)
        self.lineEdit_new_isotope = QtWidgets.QLineEdit(SetupIsotope)
        self.lineEdit_new_isotope.setObjectName("lineEdit_new_isotope")
        self.gridLayout.addWidget(self.lineEdit_new_isotope, 1, 2, 1, 1)
        self.label_seq_running_type = QtWidgets.QLabel(SetupIsotope)
        self.label_seq_running_type.setObjectName("label_seq_running_type")
        self.gridLayout.addWidget(self.label_seq_running_type, 3, 1, 1, 1)
        self.pushButton_ok = QtWidgets.QPushButton(SetupIsotope)
        self.pushButton_ok.setObjectName("pushButton_ok")
        self.gridLayout.addWidget(self.pushButton_ok, 4, 1, 1, 1)
        self.pushButton_cancel = QtWidgets.QPushButton(SetupIsotope)
        self.pushButton_cancel.setObjectName("pushButton_cancel")
        self.gridLayout.addWidget(self.pushButton_cancel, 4, 2, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)

        self.retranslateUi(SetupIsotope)
        QtCore.QMetaObject.connectSlotsByName(SetupIsotope)
        SetupIsotope.setTabOrder(self.comboBox_isotope, self.pushButton_add_new_to_db)
        SetupIsotope.setTabOrder(self.pushButton_add_new_to_db, self.lineEdit_new_isotope)
        SetupIsotope.setTabOrder(self.lineEdit_new_isotope, self.comboBox_sequencer_select)
        SetupIsotope.setTabOrder(self.comboBox_sequencer_select, self.pushButton_init_sequencer)

    def retranslateUi(self, SetupIsotope):
        _translate = QtCore.QCoreApplication.translate
        SetupIsotope.setWindowTitle(_translate("SetupIsotope", "Setup Isotope"))
        self.pushButton_add_new_to_db.setText(_translate("SetupIsotope", "add new isotope:"))
        self.pushButton_init_sequencer.setText(_translate("SetupIsotope", "initialize sequencer"))
        self.label_2.setText(_translate("SetupIsotope", "Sequencer:"))
        self.label_isotope.setText(_translate("SetupIsotope", "Isotope:"))
        self.label_sequencer_state.setText(_translate("SetupIsotope", "None"))
        self.label.setText(_translate("SetupIsotope", "status of sequencer:"))
        self.label_seq_running_type.setText(_translate("SetupIsotope", "None"))
        self.pushButton_ok.setText(_translate("SetupIsotope", "Ok"))
        self.pushButton_cancel.setText(_translate("SetupIsotope", "Cancel"))

