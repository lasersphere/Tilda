# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'E:\Users\Patrick\Documents\Python Projects\Tilda\PolliFit\source\Gui\Ui_Main.ui'
#
# Created by: PyQt5 UI code generator 5.13.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(940, 597)
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
        self.TabWidget = QtWidgets.QTabWidget(self.splitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.TabWidget.sizePolicy().hasHeightForWidth())
        self.TabWidget.setSizePolicy(sizePolicy)
        self.TabWidget.setMinimumSize(QtCore.QSize(400, 20))
        self.TabWidget.setFocusPolicy(QtCore.Qt.NoFocus)
        self.TabWidget.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.TabWidget.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.TabWidget.setObjectName("TabWidget")
        self.crawler = CrawlerUi()
        self.crawler.setObjectName("crawler")
        self.TabWidget.addTab(self.crawler, "")
        self.intfit = InteractiveFitUi()
        self.intfit.setObjectName("intfit")
        self.TabWidget.addTab(self.intfit, "")
        self.batchfit = BatchfitterUi()
        self.batchfit.setObjectName("batchfit")
        self.TabWidget.addTab(self.batchfit, "")
        self.spectrafit = SpectraFitUi()
        self.spectrafit.setStyleSheet("")
        self.spectrafit.setObjectName("spectrafit")
        self.TabWidget.addTab(self.spectrafit, "")
        self.averager = AveragerUi()
        self.averager.setObjectName("averager")
        self.TabWidget.addTab(self.averager, "")
        self.isoshift = IsoshiftUi()
        self.isoshift.setObjectName("isoshift")
        self.TabWidget.addTab(self.isoshift, "")
        self.accVolt_tab = AccVoltUi()
        self.accVolt_tab.setObjectName("accVolt_tab")
        self.TabWidget.addTab(self.accVolt_tab, "")
        self.kingfit = KingFitUi()
        self.kingfit.setObjectName("kingfit")
        self.TabWidget.addTab(self.kingfit, "")
        self.moments = MomentsUi()
        self.moments.setObjectName("moments")
        self.TabWidget.addTab(self.moments, "")
        self.addFiles_tab = AddFilesUi()
        self.addFiles_tab.setObjectName("addFiles_tab")
        self.TabWidget.addTab(self.addFiles_tab, "")
        self.Alive_tab = AliveUi()
        self.Alive_tab.setObjectName("Alive_tab")
        self.TabWidget.addTab(self.Alive_tab, "")
        self.ColAcol_tab = ColAcolUi()
        self.ColAcol_tab.setObjectName("ColAcol_tab")
        self.TabWidget.addTab(self.ColAcol_tab, "")
        self.Simulation_tab = SimulationUi()
        self.Simulation_tab.setObjectName("Simulation_tab")
        self.TabWidget.addTab(self.Simulation_tab, "")
        self.asciiConv_tab = AsciiConvUi()
        self.asciiConv_tab.setObjectName("asciiConv_tab")
        self.TabWidget.addTab(self.asciiConv_tab, "")
        self.oOut = QtWidgets.QPlainTextEdit(self.splitter)
        self.oOut.setPlainText("")
        self.oOut.setMaximumBlockCount(500)
        self.oOut.setObjectName("oOut")
        self.verticalLayout.addWidget(self.splitter)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 940, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.retranslateUi(MainWindow)
        self.TabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Database:"))
        self.oDbPath.setText(_translate("MainWindow", "TextLabel"))
        self.bOpenDb.setText(_translate("MainWindow", "Open DB"))
        self.pushButton_refresh.setText(_translate("MainWindow", "refresh"))
        self.TabWidget.setTabText(self.TabWidget.indexOf(self.crawler), _translate("MainWindow", "Crawler"))
        self.TabWidget.setTabText(self.TabWidget.indexOf(self.intfit), _translate("MainWindow", "Interactive Fit"))
        self.TabWidget.setTabText(self.TabWidget.indexOf(self.batchfit), _translate("MainWindow", "Batch Fit"))
        self.TabWidget.setTabText(self.TabWidget.indexOf(self.spectrafit), _translate("MainWindow", "Spectra Fit"))
        self.TabWidget.setTabText(self.TabWidget.indexOf(self.averager), _translate("MainWindow", "Averager"))
        self.TabWidget.setTabText(self.TabWidget.indexOf(self.isoshift), _translate("MainWindow", "Isotope shift"))
        self.TabWidget.setTabText(self.TabWidget.indexOf(self.accVolt_tab), _translate("MainWindow", "AccVolt"))
        self.TabWidget.setTabText(self.TabWidget.indexOf(self.kingfit), _translate("MainWindow", "Charge Radii"))
        self.TabWidget.setTabText(self.TabWidget.indexOf(self.moments), _translate("MainWindow", "Moments"))
        self.TabWidget.setTabText(self.TabWidget.indexOf(self.addFiles_tab), _translate("MainWindow", "add Files"))
        self.TabWidget.setTabText(self.TabWidget.indexOf(self.Alive_tab), _translate("MainWindow", "Alive"))
        self.TabWidget.setTabText(self.TabWidget.indexOf(self.ColAcol_tab), _translate("MainWindow", "Col./Acol."))
        self.TabWidget.setTabText(self.TabWidget.indexOf(self.Simulation_tab), _translate("MainWindow", "Simulation"))
        self.TabWidget.setTabText(self.TabWidget.indexOf(self.asciiConv_tab), _translate("MainWindow", "ASCII conv."))
from Gui.AccVoltUi import AccVoltUi
from Gui.AddFilesUi import AddFilesUi
from Gui.AliveUi import AliveUi
from Gui.AsciiConvUi import AsciiConvUi
from Gui.AveragerUi import AveragerUi
from Gui.BatchfitterUi import BatchfitterUi
from Gui.ColAcolUi import ColAcolUi
from Gui.CrawlerUi import CrawlerUi
from Gui.InteractiveFitUi import InteractiveFitUi
from Gui.IsoshiftUi import IsoshiftUi
from Gui.KingFitUi import KingFitUi
from Gui.MomentsUi import MomentsUi
from Gui.SimulationUi import SimulationUi
from Gui.SpectraFitUi import SpectraFitUi
