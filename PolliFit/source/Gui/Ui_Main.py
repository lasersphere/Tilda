# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_Main.ui'
#
# Created: Thu Aug  4 22:40:40 2016
#      by: PyQt5 UI code generator 5.3.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(454, 481)
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
        self.pushButton_refresh = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_refresh.setObjectName("pushButton_refresh")
        self.horizontalLayout.addWidget(self.pushButton_refresh)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.splitter = QtWidgets.QSplitter(self.centralwidget)
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.splitter.setObjectName("splitter")
        self.tabWidget_AccVoltage = QtWidgets.QTabWidget(self.splitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabWidget_AccVoltage.sizePolicy().hasHeightForWidth())
        self.tabWidget_AccVoltage.setSizePolicy(sizePolicy)
        self.tabWidget_AccVoltage.setMinimumSize(QtCore.QSize(400, 20))
        self.tabWidget_AccVoltage.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.tabWidget_AccVoltage.setObjectName("tabWidget_AccVoltage")
        self.crawler = CrawlerUi()
        self.crawler.setObjectName("crawler")
        self.tabWidget_AccVoltage.addTab(self.crawler, "")
        self.intfit = InteractiveFitUi()
        self.intfit.setObjectName("intfit")
        self.tabWidget_AccVoltage.addTab(self.intfit, "")
        self.batchfit = BatchfitterUi()
        self.batchfit.setObjectName("batchfit")
        self.tabWidget_AccVoltage.addTab(self.batchfit, "")
        self.averager = AveragerUi()
        self.averager.setObjectName("averager")
        self.tabWidget_AccVoltage.addTab(self.averager, "")
        self.isoshift = IsoshiftUi()
        self.isoshift.setObjectName("isoshift")
        self.tabWidget_AccVoltage.addTab(self.isoshift, "")
        self.accVolt_tab = AccVoltUi()
        self.accVolt_tab.setObjectName("accVolt_tab")
        self.tabWidget_AccVoltage.addTab(self.accVolt_tab, "")
        self.oOut = QtWidgets.QPlainTextEdit(self.splitter)
        self.oOut.setPlainText("")
        self.oOut.setMaximumBlockCount(500)
        self.oOut.setObjectName("oOut")
        self.verticalLayout.addWidget(self.splitter)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 454, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.retranslateUi(MainWindow)
        self.tabWidget_AccVoltage.setCurrentIndex(4)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Database:"))
        self.oDbPath.setText(_translate("MainWindow", "TextLabel"))
        self.bOpenDb.setText(_translate("MainWindow", "Open DB"))
        self.pushButton_refresh.setText(_translate("MainWindow", "refresh"))
        self.tabWidget_AccVoltage.setTabText(self.tabWidget_AccVoltage.indexOf(self.crawler), _translate("MainWindow", "Crawler"))
        self.tabWidget_AccVoltage.setTabText(self.tabWidget_AccVoltage.indexOf(self.intfit), _translate("MainWindow", "Interactive Fit"))
        self.tabWidget_AccVoltage.setTabText(self.tabWidget_AccVoltage.indexOf(self.batchfit), _translate("MainWindow", "Batch Fit"))
        self.tabWidget_AccVoltage.setTabText(self.tabWidget_AccVoltage.indexOf(self.averager), _translate("MainWindow", "Averager"))
        self.tabWidget_AccVoltage.setTabText(self.tabWidget_AccVoltage.indexOf(self.isoshift), _translate("MainWindow", "Isotope shift"))
        self.tabWidget_AccVoltage.setTabText(self.tabWidget_AccVoltage.indexOf(self.accVolt_tab), _translate("MainWindow", "AccVolt"))

from Gui.AveragerUi import AveragerUi
from Gui.InteractiveFitUi import InteractiveFitUi
from Gui.BatchfitterUi import BatchfitterUi
from Gui.CrawlerUi import CrawlerUi
from Gui.IsoshiftUi import IsoshiftUi
from Gui.AccVoltUi import AccVoltUi
