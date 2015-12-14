# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_simpleCounterRunnning.ui'
#
# Created by: PyQt5 UI code generator 5.5
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_SimpleCounterRunning(object):
    def setupUi(self, SimpleCounterRunning):
        SimpleCounterRunning.setObjectName("SimpleCounterRunning")
        SimpleCounterRunning.resize(311, 195)
        self.centralwidget = QtWidgets.QWidget(SimpleCounterRunning)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.verticalLayout_2.addLayout(self.gridLayout_2)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.comboBox_post_acc_control = QtWidgets.QComboBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox_post_acc_control.sizePolicy().hasHeightForWidth())
        self.comboBox_post_acc_control.setSizePolicy(sizePolicy)
        self.comboBox_post_acc_control.setObjectName("comboBox_post_acc_control")
        self.comboBox_post_acc_control.addItem("")
        self.comboBox_post_acc_control.addItem("")
        self.comboBox_post_acc_control.addItem("")
        self.comboBox_post_acc_control.addItem("")
        self.gridLayout.addWidget(self.comboBox_post_acc_control, 0, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.doubleSpinBox = QtWidgets.QDoubleSpinBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.doubleSpinBox.sizePolicy().hasHeightForWidth())
        self.doubleSpinBox.setSizePolicy(sizePolicy)
        self.doubleSpinBox.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.doubleSpinBox.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.doubleSpinBox.setKeyboardTracking(False)
        self.doubleSpinBox.setMinimum(-10.0)
        self.doubleSpinBox.setMaximum(10.0)
        self.doubleSpinBox.setSingleStep(0.01)
        self.doubleSpinBox.setObjectName("doubleSpinBox")
        self.gridLayout.addWidget(self.doubleSpinBox, 1, 1, 1, 1)
        self.label_dac_set_volt = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_dac_set_volt.sizePolicy().hasHeightForWidth())
        self.label_dac_set_volt.setSizePolicy(sizePolicy)
        self.label_dac_set_volt.setObjectName("label_dac_set_volt")
        self.gridLayout.addWidget(self.label_dac_set_volt, 1, 2, 1, 1)
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.pushButton_refresh_post_acc_state = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_refresh_post_acc_state.setObjectName("pushButton_refresh_post_acc_state")
        self.gridLayout_3.addWidget(self.pushButton_refresh_post_acc_state, 1, 0, 1, 1)
        self.label_post_acc_readback_state = QtWidgets.QLabel(self.centralwidget)
        self.label_post_acc_readback_state.setObjectName("label_post_acc_readback_state")
        self.gridLayout_3.addWidget(self.label_post_acc_readback_state, 0, 0, 1, 1)
        self.gridLayout.addLayout(self.gridLayout_3, 0, 2, 1, 1)
        self.verticalLayout_2.addLayout(self.gridLayout)
        self.verticalLayout.addLayout(self.verticalLayout_2)
        self.pushButton_stop = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_stop.setObjectName("pushButton_stop")
        self.verticalLayout.addWidget(self.pushButton_stop)
        SimpleCounterRunning.setCentralWidget(self.centralwidget)

        self.retranslateUi(SimpleCounterRunning)
        QtCore.QMetaObject.connectSlotsByName(SimpleCounterRunning)

    def retranslateUi(self, SimpleCounterRunning):
        _translate = QtCore.QCoreApplication.translate
        SimpleCounterRunning.setWindowTitle(_translate("SimpleCounterRunning", "Simple Counter"))
        self.label.setText(_translate("SimpleCounterRunning", "post acceleration control"))
        self.comboBox_post_acc_control.setItemText(0, _translate("SimpleCounterRunning", "Kepco"))
        self.comboBox_post_acc_control.setItemText(1, _translate("SimpleCounterRunning", "Heinzinger1"))
        self.comboBox_post_acc_control.setItemText(2, _translate("SimpleCounterRunning", "Heinzinger2"))
        self.comboBox_post_acc_control.setItemText(3, _translate("SimpleCounterRunning", "Heinzinger3"))
        self.label_2.setText(_translate("SimpleCounterRunning", "dac voltage [V]"))
        self.label_dac_set_volt.setText(_translate("SimpleCounterRunning", "TextLabel"))
        self.pushButton_refresh_post_acc_state.setText(_translate("SimpleCounterRunning", "Refresh"))
        self.label_post_acc_readback_state.setText(_translate("SimpleCounterRunning", "TextLabel"))
        self.pushButton_stop.setText(_translate("SimpleCounterRunning", "STOP"))

