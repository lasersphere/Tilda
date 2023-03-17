# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_Channel.ui'
#
# Created by: PyQt5 UI code generator 5.5
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_ChannelUi(object):
    def setupUi(self, ChannelUi):
        ChannelUi.setObjectName("ChannelUi")
        ChannelUi.resize(388, 177)
        ChannelUi.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.formLayout = QtWidgets.QFormLayout(ChannelUi)
        self.formLayout.setObjectName("formLayout")
        self.label = QtWidgets.QLabel(ChannelUi)
        self.label.setObjectName("label")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label)
        self.comboBox_out_ch = QtWidgets.QComboBox(ChannelUi)
        self.comboBox_out_ch.setObjectName("comboBox_out_ch")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.comboBox_out_ch)
        self.buttonBox = QtWidgets.QDialogButtonBox(ChannelUi)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.formLayout.setWidget(6, QtWidgets.QFormLayout.SpanningRole, self.buttonBox)
        self.label_2 = QtWidgets.QLabel(ChannelUi)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.spinBox_num_pulses = QtWidgets.QSpinBox(ChannelUi)
        self.spinBox_num_pulses.setMaximum(1000)
        self.spinBox_num_pulses.setObjectName("spinBox_num_pulses")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.spinBox_num_pulses)
        self.label_3 = QtWidgets.QLabel(ChannelUi)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.label_4 = QtWidgets.QLabel(ChannelUi)
        self.label_4.setObjectName("label_4")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.lineEdit_chan_name = QtWidgets.QLineEdit(ChannelUi)
        self.lineEdit_chan_name.setObjectName("lineEdit_chan_name")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit_chan_name)
        self.doubleSpinBox_pulse_width_us = QtWidgets.QDoubleSpinBox(ChannelUi)
        self.doubleSpinBox_pulse_width_us.setMaximum(10000000.0)
        self.doubleSpinBox_pulse_width_us.setObjectName("doubleSpinBox_pulse_width_us")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.doubleSpinBox_pulse_width_us)
        self.label_5 = QtWidgets.QLabel(ChannelUi)
        self.label_5.setObjectName("label_5")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_5)
        self.doubleSpinBox_delay_us = QtWidgets.QDoubleSpinBox(ChannelUi)
        self.doubleSpinBox_delay_us.setMaximum(10000000.0)
        self.doubleSpinBox_delay_us.setObjectName("doubleSpinBox_delay_us")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.doubleSpinBox_delay_us)
        self.label_6 = QtWidgets.QLabel(ChannelUi)
        self.label_6.setObjectName("label_6")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.comboBox_inverted = QtWidgets.QComboBox(ChannelUi)
        self.comboBox_inverted.setObjectName("comboBox_inverted")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.comboBox_inverted)

        self.retranslateUi(ChannelUi)
        self.buttonBox.accepted.connect(ChannelUi.accept)
        self.buttonBox.rejected.connect(ChannelUi.reject)
        QtCore.QMetaObject.connectSlotsByName(ChannelUi)

    def retranslateUi(self, ChannelUi):
        _translate = QtCore.QCoreApplication.translate
        ChannelUi.setWindowTitle(_translate("ChannelUi", "Dialog"))
        self.label.setText(_translate("ChannelUi", "output channel:"))
        self.label_2.setText(_translate("ChannelUi", "number of pulses:"))
        self.label_3.setText(_translate("ChannelUi", "pulse width [us]:"))
        self.label_4.setText(_translate("ChannelUi", "channel name:"))
        self.label_5.setText(_translate("ChannelUi", "delay [us]:"))
        self.label_6.setText(_translate("ChannelUi", "inverted:"))

