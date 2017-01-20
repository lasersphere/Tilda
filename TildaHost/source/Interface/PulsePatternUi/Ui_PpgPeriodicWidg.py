# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_PpgPeriodicWidg.ui'
#
# Created by: PyQt5 UI code generator 5.5
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_PpgPeriodicWidg(object):
    def setupUi(self, PpgPeriodicWidg):
        PpgPeriodicWidg.setObjectName("PpgPeriodicWidg")
        PpgPeriodicWidg.resize(377, 435)
        PpgPeriodicWidg.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.horizontalLayout = QtWidgets.QHBoxLayout(PpgPeriodicWidg)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.listWidget_periodic_pattern = QtWidgets.QListWidget(PpgPeriodicWidg)
        self.listWidget_periodic_pattern.setObjectName("listWidget_periodic_pattern")
        self.verticalLayout.addWidget(self.listWidget_periodic_pattern)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.doubleSpinBox_sys_rep_rate = QtWidgets.QDoubleSpinBox(PpgPeriodicWidg)
        self.doubleSpinBox_sys_rep_rate.setDecimals(3)
        self.doubleSpinBox_sys_rep_rate.setMaximum(10000000.0)
        self.doubleSpinBox_sys_rep_rate.setObjectName("doubleSpinBox_sys_rep_rate")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.doubleSpinBox_sys_rep_rate)
        self.label = QtWidgets.QLabel(PpgPeriodicWidg)
        self.label.setObjectName("label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.verticalLayout.addLayout(self.formLayout)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.pushButton_add_trig = QtWidgets.QPushButton(PpgPeriodicWidg)
        self.pushButton_add_trig.setObjectName("pushButton_add_trig")
        self.verticalLayout_4.addWidget(self.pushButton_add_trig)
        self.pushButton_add_ch = QtWidgets.QPushButton(PpgPeriodicWidg)
        self.pushButton_add_ch.setObjectName("pushButton_add_ch")
        self.verticalLayout_4.addWidget(self.pushButton_add_ch)
        self.pushButton_move_up = QtWidgets.QPushButton(PpgPeriodicWidg)
        self.pushButton_move_up.setObjectName("pushButton_move_up")
        self.verticalLayout_4.addWidget(self.pushButton_move_up)
        self.pushButton_move_down = QtWidgets.QPushButton(PpgPeriodicWidg)
        self.pushButton_move_down.setObjectName("pushButton_move_down")
        self.verticalLayout_4.addWidget(self.pushButton_move_down)
        self.pushButton_edit_selected = QtWidgets.QPushButton(PpgPeriodicWidg)
        self.pushButton_edit_selected.setObjectName("pushButton_edit_selected")
        self.verticalLayout_4.addWidget(self.pushButton_edit_selected)
        self.pushButton_remove_selected_periodic = QtWidgets.QPushButton(PpgPeriodicWidg)
        self.pushButton_remove_selected_periodic.setObjectName("pushButton_remove_selected_periodic")
        self.verticalLayout_4.addWidget(self.pushButton_remove_selected_periodic)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_4.addItem(spacerItem)
        self.horizontalLayout.addLayout(self.verticalLayout_4)

        self.retranslateUi(PpgPeriodicWidg)
        QtCore.QMetaObject.connectSlotsByName(PpgPeriodicWidg)

    def retranslateUi(self, PpgPeriodicWidg):
        _translate = QtCore.QCoreApplication.translate
        PpgPeriodicWidg.setWindowTitle(_translate("PpgPeriodicWidg", "Form"))
        self.label.setText(_translate("PpgPeriodicWidg", "system repetition time [us]:"))
        self.pushButton_add_trig.setText(_translate("PpgPeriodicWidg", "add trigger"))
        self.pushButton_add_ch.setText(_translate("PpgPeriodicWidg", "add channel"))
        self.pushButton_move_up.setText(_translate("PpgPeriodicWidg", "move up"))
        self.pushButton_move_down.setText(_translate("PpgPeriodicWidg", "move down"))
        self.pushButton_edit_selected.setText(_translate("PpgPeriodicWidg", "edit selected"))
        self.pushButton_remove_selected_periodic.setText(_translate("PpgPeriodicWidg", "remove selected"))

