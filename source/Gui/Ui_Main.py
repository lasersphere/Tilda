# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_Main.ui'
#
# Created: Fri Jun  6 17:03:52 2014
#      by: PyQt5 UI code generator 5.2.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(714, 547)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setMinimumSize(QtCore.QSize(200, 0))
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout.addWidget(self.pushButton)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.splitter = QtWidgets.QSplitter(self.centralwidget)
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.splitter.setObjectName("splitter")
        self.tabWidget = QtWidgets.QTabWidget(self.splitter)
        self.tabWidget.setMinimumSize(QtCore.QSize(0, 200))
        self.tabWidget.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.tabWidget.setObjectName("tabWidget")
        self.crawler = CrawlerUi()
        self.crawler.setObjectName("crawler")
        self.tabWidget.addTab(self.crawler, "")
        self.fitter = QtWidgets.QWidget()
        self.fitter.setObjectName("fitter")
        self.tabWidget.addTab(self.fitter, "")
        self.averager = QtWidgets.QWidget()
        self.averager.setObjectName("averager")
        self.tabWidget.addTab(self.averager, "")
        self.shift = QtWidgets.QWidget()
        self.shift.setObjectName("shift")
        self.tabWidget.addTab(self.shift, "")
        self.summary = QtWidgets.QWidget()
        self.summary.setObjectName("summary")
        self.tabWidget.addTab(self.summary, "")
        self.plainTextEdit = QtWidgets.QPlainTextEdit(self.splitter)
        self.plainTextEdit.setObjectName("plainTextEdit")
        self.verticalLayout.addWidget(self.splitter)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 714, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Database:"))
        self.label_2.setText(_translate("MainWindow", "TextLabel"))
        self.pushButton.setText(_translate("MainWindow", "Open DB"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.crawler), _translate("MainWindow", "Crawler"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.fitter), _translate("MainWindow", "Fitter"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.averager), _translate("MainWindow", "Averager"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.shift), _translate("MainWindow", "Shift"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.summary), _translate("MainWindow", "Summary"))

from Gui.CrawlerUi import CrawlerUi
