# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_SingleHit.ui'
#
# Created by: PyQt5 UI code generator 5.5
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_single_hit_widg(object):
    def setupUi(self, single_hit_widg):
        single_hit_widg.setObjectName("single_hit_widg")
        single_hit_widg.resize(385, 64)
        single_hit_widg.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.gridLayout = QtWidgets.QGridLayout(single_hit_widg)
        self.gridLayout.setObjectName("gridLayout")
        self.comboBox_trigInputChan = QtWidgets.QComboBox(single_hit_widg)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox_trigInputChan.sizePolicy().hasHeightForWidth())
        self.comboBox_trigInputChan.setSizePolicy(sizePolicy)
        self.comboBox_trigInputChan.setObjectName("comboBox_trigInputChan")
        self.gridLayout.addWidget(self.comboBox_trigInputChan, 0, 1, 1, 1)
        self.label = QtWidgets.QLabel(single_hit_widg)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.label_trigInputChan_set = QtWidgets.QLabel(single_hit_widg)
        self.label_trigInputChan_set.setObjectName("label_trigInputChan_set")
        self.gridLayout.addWidget(self.label_trigInputChan_set, 0, 2, 1, 1)
        self.label_3 = QtWidgets.QLabel(single_hit_widg)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 1, 0, 1, 1)
        self.comboBox_trigger_edge = QtWidgets.QComboBox(single_hit_widg)
        self.comboBox_trigger_edge.setObjectName("comboBox_trigger_edge")
        self.gridLayout.addWidget(self.comboBox_trigger_edge, 1, 1, 1, 1)
        self.label_selected_trigger_edge = QtWidgets.QLabel(single_hit_widg)
        self.label_selected_trigger_edge.setObjectName("label_selected_trigger_edge")
        self.gridLayout.addWidget(self.label_selected_trigger_edge, 1, 2, 1, 1)

        self.retranslateUi(single_hit_widg)
        QtCore.QMetaObject.connectSlotsByName(single_hit_widg)

    def retranslateUi(self, single_hit_widg):
        _translate = QtCore.QCoreApplication.translate
        single_hit_widg.setWindowTitle(_translate("single_hit_widg", "single_hit_widg"))
        self.label.setText(_translate("single_hit_widg", "trigger input channel"))
        self.label_trigInputChan_set.setText(_translate("single_hit_widg", "TextLabel"))
        self.label_3.setText(_translate("single_hit_widg", "trigger edge"))
        self.label_selected_trigger_edge.setText(_translate("single_hit_widg", "TextLabel"))

