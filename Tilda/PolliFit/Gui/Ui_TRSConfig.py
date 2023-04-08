# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'E:\Users\Patrick\Documents\Python Projects\Tilda\PolliFit\source\Gui\Ui_TRSConfig.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_TRSConfig(object):
    def setupUi(self, TRSConfig):
        TRSConfig.setObjectName("TRSConfig")
        TRSConfig.resize(400, 200)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(TRSConfig.sizePolicy().hasHeightForWidth())
        TRSConfig.setSizePolicy(sizePolicy)
        TRSConfig.setMinimumSize(QtCore.QSize(0, 200))
        self.verticalLayout = QtWidgets.QVBoxLayout(TRSConfig)
        self.verticalLayout.setSpacing(10)
        self.verticalLayout.setObjectName("verticalLayout")
        self.l_info = QtWidgets.QLabel(TRSConfig)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.l_info.setFont(font)
        self.l_info.setObjectName("l_info")
        self.verticalLayout.addWidget(self.l_info)
        spacerItem = QtWidgets.QSpacerItem(20, 6, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout.addItem(spacerItem)
        self.grid_trs = QtWidgets.QGridLayout()
        self.grid_trs.setObjectName("grid_trs")
        self.verticalLayout.addLayout(self.grid_trs)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem1)
        self.hor_buttons = QtWidgets.QHBoxLayout()
        self.hor_buttons.setObjectName("hor_buttons")
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.hor_buttons.addItem(spacerItem2)
        self.b_ok = QtWidgets.QPushButton(TRSConfig)
        self.b_ok.setObjectName("b_ok")
        self.hor_buttons.addWidget(self.b_ok)
        self.b_cancel = QtWidgets.QPushButton(TRSConfig)
        self.b_cancel.setObjectName("b_cancel")
        self.hor_buttons.addWidget(self.b_cancel)
        self.verticalLayout.addLayout(self.hor_buttons)
        self.verticalLayout.setStretch(3, 1)

        self.retranslateUi(TRSConfig)
        QtCore.QMetaObject.connectSlotsByName(TRSConfig)

    def retranslateUi(self, TRSConfig):
        _translate = QtCore.QCoreApplication.translate
        TRSConfig.setWindowTitle(_translate("TRSConfig", "Time-Resolved Spectra Configuration"))
        self.l_info.setText(_translate("TRSConfig", "Software gates <nobr>[t<sub>min</sub> , t<sub>max</sub>] (&mu;s)</nobr>"))
        self.b_ok.setText(_translate("TRSConfig", "OK"))
        self.b_cancel.setText(_translate("TRSConfig", "Cancel"))
