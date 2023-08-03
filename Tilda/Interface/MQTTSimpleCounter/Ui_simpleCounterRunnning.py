# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\Edwar\OneDrive\Desktop\PhD 2023\Tilda_Improvements\UserInterfaces\Ui_simpleCounterRunnning.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_SimpleCounterRunning(object):
    def setupUi(self, SimpleCounterRunning):
        SimpleCounterRunning.setObjectName("SimpleCounterRunning")
        SimpleCounterRunning.resize(281, 89)
        SimpleCounterRunning.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.centralwidget = QtWidgets.QWidget(SimpleCounterRunning)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.hor_control = QtWidgets.QHBoxLayout()
        self.hor_control.setObjectName("hor_control")
        self.grid_control = QtWidgets.QGridLayout()
        self.grid_control.setContentsMargins(-1, -1, 0, -1)
        self.grid_control.setHorizontalSpacing(20)
        self.grid_control.setObjectName("grid_control")
        self.label_dac = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_dac.sizePolicy().hasHeightForWidth())
        self.label_dac.setSizePolicy(sizePolicy)
        self.label_dac.setMinimumSize(QtCore.QSize(0, 25))
        self.label_dac.setObjectName("label_dac")
        self.grid_control.addWidget(self.label_dac, 0, 2, 1, 1)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.doubleSpinBox = QtWidgets.QDoubleSpinBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.doubleSpinBox.sizePolicy().hasHeightForWidth())
        self.doubleSpinBox.setSizePolicy(sizePolicy)
        self.doubleSpinBox.setMinimumSize(QtCore.QSize(60, 0))
        self.doubleSpinBox.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.doubleSpinBox.setReadOnly(True)
        self.doubleSpinBox.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.doubleSpinBox.setKeyboardTracking(False)
        self.doubleSpinBox.setDecimals(3)
        self.doubleSpinBox.setMinimum(-10.0)
        self.doubleSpinBox.setMaximum(10.0)
        self.doubleSpinBox.setSingleStep(0.001)
        self.doubleSpinBox.setProperty("value", 0.0)
        self.doubleSpinBox.setObjectName("doubleSpinBox")
        self.horizontalLayout_3.addWidget(self.doubleSpinBox)
        self.grid_control.addLayout(self.horizontalLayout_3, 1, 2, 1, 1)
        self.checkBox = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox.setChecked(False)
        self.checkBox.setObjectName("checkBox")
        self.grid_control.addWidget(self.checkBox, 0, 1, 2, 1)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.grid_control.addItem(spacerItem, 0, 0, 2, 1)
        self.hor_control.addLayout(self.grid_control)
        self.verticalLayout.addLayout(self.hor_control)
        SimpleCounterRunning.setCentralWidget(self.centralwidget)

        self.retranslateUi(SimpleCounterRunning)
        QtCore.QMetaObject.connectSlotsByName(SimpleCounterRunning)

    def retranslateUi(self, SimpleCounterRunning):
        _translate = QtCore.QCoreApplication.translate
        SimpleCounterRunning.setWindowTitle(_translate("SimpleCounterRunning", "Simple Counter"))
        self.label_dac.setText(_translate("SimpleCounterRunning", "DAC voltage"))
        self.doubleSpinBox.setSuffix(_translate("SimpleCounterRunning", " V"))
        self.checkBox.setText(_translate("SimpleCounterRunning", "Receiving MQTT"))

