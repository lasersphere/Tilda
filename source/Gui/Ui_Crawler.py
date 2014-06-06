# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_Crawler.ui'
#
# Created: Fri Jun  6 17:00:36 2014
#      by: PyQt5 UI code generator 5.2.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Crawler(object):
    def setupUi(self, Crawler):
        Crawler.setObjectName("Crawler")
        Crawler.resize(400, 300)
        self.pushButton = QtWidgets.QPushButton(Crawler)
        self.pushButton.setGeometry(QtCore.QRect(160, 130, 75, 23))
        self.pushButton.setObjectName("pushButton")

        self.retranslateUi(Crawler)
        QtCore.QMetaObject.connectSlotsByName(Crawler)

    def retranslateUi(self, Crawler):
        _translate = QtCore.QCoreApplication.translate
        Crawler.setWindowTitle(_translate("Crawler", "Form"))
        self.pushButton.setText(_translate("Crawler", "PushButton"))

