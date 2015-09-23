# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_Version.ui'
#
# Created: Wed Sep 23 15:11:59 2015
#      by: PyQt5 UI code generator 5.3.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Version(object):
    def setupUi(self, Version):
        Version.setObjectName("Version")
        Version.resize(188, 81)
        self.gridLayout_2 = QtWidgets.QGridLayout(Version)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.labelDateFix = QtWidgets.QLabel(Version)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labelDateFix.sizePolicy().hasHeightForWidth())
        self.labelDateFix.setSizePolicy(sizePolicy)
        self.labelDateFix.setObjectName("labelDateFix")
        self.gridLayout.addWidget(self.labelDateFix, 1, 0, 1, 1)
        self.labelDate = QtWidgets.QLabel(Version)
        self.labelDate.setObjectName("labelDate")
        self.gridLayout.addWidget(self.labelDate, 1, 1, 1, 1)
        self.labelVersion = QtWidgets.QLabel(Version)
        self.labelVersion.setObjectName("labelVersion")
        self.gridLayout.addWidget(self.labelVersion, 0, 1, 1, 1)
        self.labelVersionFix = QtWidgets.QLabel(Version)
        self.labelVersionFix.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labelVersionFix.sizePolicy().hasHeightForWidth())
        self.labelVersionFix.setSizePolicy(sizePolicy)
        self.labelVersionFix.setObjectName("labelVersionFix")
        self.gridLayout.addWidget(self.labelVersionFix, 0, 0, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)

        self.retranslateUi(Version)
        QtCore.QMetaObject.connectSlotsByName(Version)

    def retranslateUi(self, Version):
        _translate = QtCore.QCoreApplication.translate
        Version.setWindowTitle(_translate("Version", "Version"))
        self.labelDateFix.setText(_translate("Version", "Date:"))
        self.labelDate.setText(_translate("Version", "TextLabel"))
        self.labelVersion.setText(_translate("Version", "TextLabel"))
        self.labelVersionFix.setText(_translate("Version", "Version:"))

