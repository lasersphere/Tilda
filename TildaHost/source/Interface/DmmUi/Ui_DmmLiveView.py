# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_DmmLiveView.ui'
#
# Created by: PyQt5 UI code generator 5.5
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(409, 262)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.tab_0 = QtWidgets.QWidget()
        self.tab_0.setObjectName("tab_0")
        self.tabWidget.addTab(self.tab_0, "")
        self.verticalLayout.addWidget(self.tabWidget)
        self.pushButton_confirm = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_confirm.setObjectName("pushButton_confirm")
        self.verticalLayout.addWidget(self.pushButton_confirm)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_0), _translate("MainWindow", "Tab 1"))
        self.pushButton_confirm.setText(_translate("MainWindow", "ok"))

