# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_Crawler.ui'
#
# Created: Tue Jun 10 14:27:33 2014
#      by: PyQt5 UI code generator 5.2.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Crawler(object):
    def setupUi(self, Crawler):
        Crawler.setObjectName("Crawler")
        Crawler.resize(400, 300)
        self.verticalLayout = QtWidgets.QVBoxLayout(Crawler)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(Crawler)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setMinimumSize(QtCore.QSize(80, 0))
        self.label.setMaximumSize(QtCore.QSize(80, 16777215))
        self.label.setWordWrap(True)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.path = QtWidgets.QLineEdit(Crawler)
        self.path.setObjectName("path")
        self.horizontalLayout.addWidget(self.path)
        self.recursive = QtWidgets.QCheckBox(Crawler)
        self.recursive.setChecked(True)
        self.recursive.setObjectName("recursive")
        self.horizontalLayout.addWidget(self.recursive)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.bcrawl = QtWidgets.QPushButton(Crawler)
        self.bcrawl.setObjectName("bcrawl")
        self.verticalLayout.addWidget(self.bcrawl)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)

        self.retranslateUi(Crawler)
        QtCore.QMetaObject.connectSlotsByName(Crawler)

    def retranslateUi(self, Crawler):
        _translate = QtCore.QCoreApplication.translate
        Crawler.setWindowTitle(_translate("Crawler", "Form"))
        self.label.setText(_translate("Crawler", "Data Folder (relative to DB)"))
        self.recursive.setText(_translate("Crawler", "Recursive"))
        self.bcrawl.setText(_translate("Crawler", "Crawl"))

