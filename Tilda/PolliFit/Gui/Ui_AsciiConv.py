# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'E:\Users\Patrick\Documents\Python Projects\Tilda\PolliFit\source\Gui\Ui_AsciiConv.ui'
#
# Created by: PyQt5 UI code generator 5.13.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_AsciiConv(object):
    def setupUi(self, AsciiConv):
        AsciiConv.setObjectName("AsciiConv")
        AsciiConv.resize(403, 306)
        self.verticalLayout = QtWidgets.QVBoxLayout(AsciiConv)
        self.verticalLayout.setObjectName("verticalLayout")
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.pushButton_open_out_dir = QtWidgets.QPushButton(AsciiConv)
        self.pushButton_open_out_dir.setObjectName("pushButton_open_out_dir")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.pushButton_open_out_dir)
        self.pushButton_choose_output = QtWidgets.QPushButton(AsciiConv)
        self.pushButton_choose_output.setObjectName("pushButton_choose_output")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.pushButton_choose_output)
        self.label_7 = QtWidgets.QLabel(AsciiConv)
        self.label_7.setObjectName("label_7")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_7)
        self.label_current_output = QtWidgets.QLabel(AsciiConv)
        self.label_current_output.setObjectName("label_current_output")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.label_current_output)
        self.label_6 = QtWidgets.QLabel(AsciiConv)
        self.label_6.setObjectName("label_6")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.lineEdit_add_name = QtWidgets.QLineEdit(AsciiConv)
        self.lineEdit_add_name.setObjectName("lineEdit_add_name")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.lineEdit_add_name)
        self.label = QtWidgets.QLabel(AsciiConv)
        self.label.setObjectName("label")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label)
        self.label_3 = QtWidgets.QLabel(AsciiConv)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.label_2 = QtWidgets.QLabel(AsciiConv)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.label_4 = QtWidgets.QLabel(AsciiConv)
        self.label_4.setObjectName("label_4")
        self.formLayout.setWidget(7, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.label_5 = QtWidgets.QLabel(AsciiConv)
        self.label_5.setObjectName("label_5")
        self.formLayout.setWidget(8, QtWidgets.QFormLayout.LabelRole, self.label_5)
        self.checkBox_x_axis_in_freq = QtWidgets.QCheckBox(AsciiConv)
        self.checkBox_x_axis_in_freq.setObjectName("checkBox_x_axis_in_freq")
        self.formLayout.setWidget(8, QtWidgets.QFormLayout.FieldRole, self.checkBox_x_axis_in_freq)
        self.hor_track_layout = QtWidgets.QHBoxLayout()
        self.hor_track_layout.setObjectName("hor_track_layout")
        self.lineEdit_tracks = QtWidgets.QSpinBox(AsciiConv)
        self.lineEdit_tracks.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.lineEdit_tracks.setCorrectionMode(QtWidgets.QAbstractSpinBox.CorrectToNearestValue)
        self.lineEdit_tracks.setMinimum(-1)
        self.lineEdit_tracks.setProperty("value", -1)
        self.lineEdit_tracks.setObjectName("lineEdit_tracks")
        self.hor_track_layout.addWidget(self.lineEdit_tracks)
        self.info_tracks = QtWidgets.QLabel(AsciiConv)
        self.info_tracks.setObjectName("info_tracks")
        self.hor_track_layout.addWidget(self.info_tracks)
        self.hor_track_layout.setStretch(0, 1)
        self.formLayout.setLayout(5, QtWidgets.QFormLayout.FieldRole, self.hor_track_layout)
        self.hor_softw_gates_layout = QtWidgets.QHBoxLayout()
        self.hor_softw_gates_layout.setObjectName("hor_softw_gates_layout")
        self.lineEdit_softw_gates = QtWidgets.QLineEdit(AsciiConv)
        self.lineEdit_softw_gates.setObjectName("lineEdit_softw_gates")
        self.hor_softw_gates_layout.addWidget(self.lineEdit_softw_gates)
        self.info_softw_gates = QtWidgets.QLabel(AsciiConv)
        self.info_softw_gates.setObjectName("info_softw_gates")
        self.hor_softw_gates_layout.addWidget(self.info_softw_gates)
        self.hor_softw_gates_layout.setStretch(0, 1)
        self.formLayout.setLayout(6, QtWidgets.QFormLayout.FieldRole, self.hor_softw_gates_layout)
        self.lineEdit_lineVar = QtWidgets.QComboBox(AsciiConv)
        self.lineEdit_lineVar.setObjectName("lineEdit_lineVar")
        self.formLayout.setWidget(7, QtWidgets.QFormLayout.FieldRole, self.lineEdit_lineVar)
        self.hor_scalers_layout = QtWidgets.QHBoxLayout()
        self.hor_scalers_layout.setObjectName("hor_scalers_layout")
        self.lineEdit_scalers = QtWidgets.QLineEdit(AsciiConv)
        self.lineEdit_scalers.setObjectName("lineEdit_scalers")
        self.hor_scalers_layout.addWidget(self.lineEdit_scalers)
        self.sc_minus = QtWidgets.QPushButton(AsciiConv)
        self.sc_minus.setMaximumSize(QtCore.QSize(60, 16777215))
        self.sc_minus.setObjectName("sc_minus")
        self.hor_scalers_layout.addWidget(self.sc_minus)
        self.sc_plus = QtWidgets.QPushButton(AsciiConv)
        self.sc_plus.setMaximumSize(QtCore.QSize(60, 16777215))
        self.sc_plus.setObjectName("sc_plus")
        self.hor_scalers_layout.addWidget(self.sc_plus)
        self.hor_scalers_layout.setStretch(0, 1)
        self.formLayout.setLayout(4, QtWidgets.QFormLayout.FieldRole, self.hor_scalers_layout)
        self.verticalLayout.addLayout(self.formLayout)
        self.pushButton_sel_and_conv = QtWidgets.QPushButton(AsciiConv)
        self.pushButton_sel_and_conv.setObjectName("pushButton_sel_and_conv")
        self.verticalLayout.addWidget(self.pushButton_sel_and_conv)

        self.retranslateUi(AsciiConv)
        QtCore.QMetaObject.connectSlotsByName(AsciiConv)

    def retranslateUi(self, AsciiConv):
        _translate = QtCore.QCoreApplication.translate
        AsciiConv.setWindowTitle(_translate("AsciiConv", "Form"))
        self.pushButton_open_out_dir.setText(_translate("AsciiConv", "open output dir"))
        self.pushButton_choose_output.setText(_translate("AsciiConv", "choose output"))
        self.label_7.setText(_translate("AsciiConv", "selected output:"))
        self.label_current_output.setText(_translate("AsciiConv", "None"))
        self.label_6.setText(_translate("AsciiConv", "add to name:"))
        self.label.setText(_translate("AsciiConv", "scalers:"))
        self.label_3.setText(_translate("AsciiConv", "track index:"))
        self.label_2.setText(_translate("AsciiConv", "software gates:"))
        self.label_4.setText(_translate("AsciiConv", "lineVar:"))
        self.label_5.setText(_translate("AsciiConv", "x in frequency"))
        self.checkBox_x_axis_in_freq.setText(_translate("AsciiConv", "x in line volts"))
        self.info_tracks.setText(_translate("AsciiConv", "Set to -1 to use all tracks"))
        self.info_softw_gates.setText(_translate("AsciiConv", "shape: (track, scaler, gates)"))
        self.sc_minus.setText(_translate("AsciiConv", "-"))
        self.sc_plus.setText(_translate("AsciiConv", "+"))
        self.pushButton_sel_and_conv.setText(_translate("AsciiConv", "select and convert files"))
