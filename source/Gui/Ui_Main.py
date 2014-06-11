# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_Main.ui'
#
# Created: Wed Jun 11 17:50:09 2014
#      by: PyQt5 UI code generator 5.2.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(714, 543)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.oDbPath = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.oDbPath.sizePolicy().hasHeightForWidth())
        self.oDbPath.setSizePolicy(sizePolicy)
        self.oDbPath.setMinimumSize(QtCore.QSize(200, 0))
        self.oDbPath.setObjectName("oDbPath")
        self.horizontalLayout.addWidget(self.oDbPath)
        self.bOpenDb = QtWidgets.QPushButton(self.centralwidget)
        self.bOpenDb.setObjectName("bOpenDb")
        self.horizontalLayout.addWidget(self.bOpenDb)
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
        self.intfit = InteractiveFitUi()
        self.intfit.setObjectName("intfit")
        self.tabWidget.addTab(self.intfit, "")
        self.batchfit = QtWidgets.QWidget()
        self.batchfit.setObjectName("batchfit")
        self.tabWidget.addTab(self.batchfit, "")
        self.averager = AveragerUi()
        self.averager.setObjectName("averager")
        self.tabWidget.addTab(self.averager, "")
        self.shift = QtWidgets.QWidget()
        self.shift.setObjectName("shift")
        self.tabWidget.addTab(self.shift, "")
        self.summary = QtWidgets.QWidget()
        self.summary.setObjectName("summary")
        self.tabWidget.addTab(self.summary, "")
        self.oOut = QtWidgets.QPlainTextEdit(self.splitter)
        self.oOut.setPlainText("")
        self.oOut.setMaximumBlockCount(500)
        self.oOut.setObjectName("oOut")
        self.verticalLayout.addWidget(self.splitter)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 714, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(3)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Database:"))
        self.oDbPath.setText(_translate("MainWindow", "TextLabel"))
        self.bOpenDb.setText(_translate("MainWindow", "Open DB"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.crawler), _translate("MainWindow", "Crawler"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.intfit), _translate("MainWindow", "Interactive Fit"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.batchfit), _translate("MainWindow", "Batch Fit"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.averager), _translate("MainWindow", "Averager"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.shift), _translate("MainWindow", "Shift"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.summary), _translate("MainWindow", "Summary"))

from Gui.CrawlerUi import CrawlerUi
from Gui.InteractiveFitUi import InteractiveFitUi
from Gui.AveragerUi import AveragerUi
