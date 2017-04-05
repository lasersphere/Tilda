# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_Main.ui'
#
# Created by: PyQt5 UI code generator 5.5
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(651, 481)
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
        self.asciiconverter = QtWidgets.QTabWidget(self.splitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.asciiconverter.sizePolicy().hasHeightForWidth())
        self.asciiconverter.setSizePolicy(sizePolicy)
        self.asciiconverter.setMinimumSize(QtCore.QSize(400, 20))
        self.asciiconverter.setFocusPolicy(QtCore.Qt.NoFocus)
        self.asciiconverter.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.asciiconverter.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.asciiconverter.setObjectName("asciiconverter")
        self.crawler = CrawlerUi()
        self.crawler.setObjectName("crawler")
        self.asciiconverter.addTab(self.crawler, "")
        self.intfit = InteractiveFitUi()
        self.intfit.setObjectName("intfit")
        self.asciiconverter.addTab(self.intfit, "")
        self.batchfit = BatchfitterUi()
        self.batchfit.setObjectName("batchfit")
        self.asciiconverter.addTab(self.batchfit, "")
        self.averager = AveragerUi()
        self.averager.setObjectName("averager")
        self.asciiconverter.addTab(self.averager, "")
        self.isoshift = IsoshiftUi()
        self.isoshift.setObjectName("isoshift")
        self.asciiconverter.addTab(self.isoshift, "")
        self.accVolt_tab = AccVoltUi()
        self.accVolt_tab.setObjectName("accVolt_tab")
        self.asciiconverter.addTab(self.accVolt_tab, "")
        self.kingfit = KingFitUi()
        self.kingfit.setObjectName("kingfit")
        self.asciiconverter.addTab(self.kingfit, "")
        self.addFiles_tab = AddFilesUi()
        self.addFiles_tab.setObjectName("addFiles_tab")
        self.asciiconverter.addTab(self.addFiles_tab, "")
        self.Alive_tab = AliveUi()
        self.Alive_tab.setObjectName("Alive_tab")
        self.asciiconverter.addTab(self.Alive_tab, "")
        self.asciiConv_tab = AsciiConvUi()
        self.asciiConv_tab.setObjectName("asciiConv_tab")
        self.asciiconverter.addTab(self.asciiConv_tab, "")
        self.oOut = QtWidgets.QPlainTextEdit(self.splitter)
        self.oOut.setPlainText("")
        self.oOut.setMaximumBlockCount(500)
        self.oOut.setObjectName("oOut")
        self.verticalLayout.addWidget(self.splitter)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 651, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.retranslateUi(MainWindow)
        self.asciiconverter.setCurrentIndex(9)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Database:"))
        self.oDbPath.setText(_translate("MainWindow", "TextLabel"))
        self.bOpenDb.setText(_translate("MainWindow", "Open DB"))
        self.pushButton_refresh.setText(_translate("MainWindow", "refresh"))
        self.asciiconverter.setTabText(self.asciiconverter.indexOf(self.crawler), _translate("MainWindow", "Crawler"))
        self.asciiconverter.setTabText(self.asciiconverter.indexOf(self.intfit), _translate("MainWindow", "Interactive Fit"))
        self.asciiconverter.setTabText(self.asciiconverter.indexOf(self.batchfit), _translate("MainWindow", "Batch Fit"))
        self.asciiconverter.setTabText(self.asciiconverter.indexOf(self.averager), _translate("MainWindow", "Averager"))
        self.asciiconverter.setTabText(self.asciiconverter.indexOf(self.isoshift), _translate("MainWindow", "Isotope shift"))
        self.asciiconverter.setTabText(self.asciiconverter.indexOf(self.accVolt_tab), _translate("MainWindow", "AccVolt"))
        self.asciiconverter.setTabText(self.asciiconverter.indexOf(self.kingfit), _translate("MainWindow", "Charge Radii"))
        self.asciiconverter.setTabText(self.asciiconverter.indexOf(self.addFiles_tab), _translate("MainWindow", "add Files"))
        self.asciiconverter.setTabText(self.asciiconverter.indexOf(self.Alive_tab), _translate("MainWindow", "Alive"))
        self.asciiconverter.setTabText(self.asciiconverter.indexOf(self.asciiConv_tab), _translate("MainWindow", "ASCII"))

from Gui.AccVoltUi import AccVoltUi
from Gui.AddFilesUi import AddFilesUi
from Gui.AliveUi import AliveUi
from Gui.AsciiConvUi import AsciiConvUi
from Gui.AveragerUi import AveragerUi
from Gui.BatchfitterUi import BatchfitterUi
from Gui.CrawlerUi import CrawlerUi
from Gui.InteractiveFitUi import InteractiveFitUi
from Gui.IsoshiftUi import IsoshiftUi
from Gui.KingFitUi import KingFitUi
