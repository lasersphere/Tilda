# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_PulsePattern.ui'
#
# Created by: PyQt5 UI code generator 5.5
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtWidgets

class Ui_PulsePatternWin(object):
    def setupUi(self, PulsePatternWin):
        PulsePatternWin.setObjectName("PulsePatternWin")
        PulsePatternWin.resize(1220, 548)
        PulsePatternWin.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.centralwidget = QtWidgets.QWidget(PulsePatternWin)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.splitter = QtWidgets.QSplitter(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.splitter.sizePolicy().hasHeightForWidth())
        self.splitter.setSizePolicy(sizePolicy)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.widget_graph_view = QtWidgets.QWidget(self.splitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_graph_view.sizePolicy().hasHeightForWidth())
        self.widget_graph_view.setSizePolicy(sizePolicy)
        self.widget_graph_view.setObjectName("widget_graph_view")
        self.tabWidget_periodic_pattern = QtWidgets.QTabWidget(self.splitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabWidget_periodic_pattern.sizePolicy().hasHeightForWidth())
        self.tabWidget_periodic_pattern.setSizePolicy(sizePolicy)
        self.tabWidget_periodic_pattern.setObjectName("tabWidget_periodic_pattern")
        self.tab_list_view = QtWidgets.QWidget()
        self.tab_list_view.setObjectName("tab_list_view")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.tab_list_view)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.listWidget_cmd_list = QtWidgets.QListWidget(self.tab_list_view)
        self.listWidget_cmd_list.setObjectName("listWidget_cmd_list")
        self.horizontalLayout_2.addWidget(self.listWidget_cmd_list)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.pushButton_add_cmd = QtWidgets.QPushButton(self.tab_list_view)
        self.pushButton_add_cmd.setObjectName("pushButton_add_cmd")
        self.verticalLayout_2.addWidget(self.pushButton_add_cmd)
        self.pushButton_remove_selected = QtWidgets.QPushButton(self.tab_list_view)
        self.pushButton_remove_selected.setObjectName("pushButton_remove_selected")
        self.verticalLayout_2.addWidget(self.pushButton_remove_selected)
        self.pushButton_load_txt = QtWidgets.QPushButton(self.tab_list_view)
        self.pushButton_load_txt.setObjectName("pushButton_load_txt")
        self.verticalLayout_2.addWidget(self.pushButton_load_txt)
        self.pushButton_save_txt = QtWidgets.QPushButton(self.tab_list_view)
        self.pushButton_save_txt.setObjectName("pushButton_save_txt")
        self.verticalLayout_2.addWidget(self.pushButton_save_txt)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.tabWidget_periodic_pattern.addTab(self.tab_list_view, "")
        self.tab_periodic_pattern = QtWidgets.QWidget()
        self.tab_periodic_pattern.setObjectName("tab_periodic_pattern")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.tab_periodic_pattern)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.tabWidget_periodic_pattern.addTab(self.tab_periodic_pattern, "")
        self.tab_simple = QtWidgets.QWidget()
        self.tab_simple.setObjectName("tab_simple")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.tab_simple)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.tabWidget_periodic_pattern.addTab(self.tab_simple, "")
        self.verticalLayout.addWidget(self.splitter)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_reset_fpga = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_reset_fpga.setObjectName("pushButton_reset_fpga")
        self.horizontalLayout.addWidget(self.pushButton_reset_fpga)
        self.pushButton_stop = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_stop.setObjectName("pushButton_stop")
        self.horizontalLayout.addWidget(self.pushButton_stop)
        self.pushButton_run_pattern = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_run_pattern.setObjectName("pushButton_run_pattern")
        self.horizontalLayout.addWidget(self.pushButton_run_pattern)
        self.pushButton_close = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_close.setObjectName("pushButton_close")
        self.horizontalLayout.addWidget(self.pushButton_close)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_3.addWidget(self.label_2)
        self.label_ppg_state = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_ppg_state.sizePolicy().hasHeightForWidth())
        self.label_ppg_state.setSizePolicy(sizePolicy)
        self.label_ppg_state.setObjectName("label_ppg_state")
        self.horizontalLayout_3.addWidget(self.label_ppg_state)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        PulsePatternWin.setCentralWidget(self.centralwidget)
        self.menuBar = QtWidgets.QMenuBar(PulsePatternWin)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 1220, 21))
        self.menuBar.setObjectName("menuBar")
        self.menuHelp = QtWidgets.QMenu(self.menuBar)
        self.menuHelp.setObjectName("menuHelp")
        PulsePatternWin.setMenuBar(self.menuBar)
        self.actionHelp = QtWidgets.QAction(PulsePatternWin)
        self.actionHelp.setObjectName("actionHelp")
        self.menuHelp.addAction(self.actionHelp)
        self.menuBar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(PulsePatternWin)
        self.tabWidget_periodic_pattern.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(PulsePatternWin)

    def retranslateUi(self, PulsePatternWin):
        _translate = QtCore.QCoreApplication.translate
        PulsePatternWin.setWindowTitle(_translate("PulsePatternWin", "MainWindow"))
        self.pushButton_add_cmd.setText(_translate("PulsePatternWin", "add cmd"))
        self.pushButton_remove_selected.setText(_translate("PulsePatternWin", "rem sel."))
        self.pushButton_load_txt.setText(_translate("PulsePatternWin", "load .txt"))
        self.pushButton_save_txt.setText(_translate("PulsePatternWin", "save .txt"))
        self.tabWidget_periodic_pattern.setTabText(self.tabWidget_periodic_pattern.indexOf(self.tab_list_view), _translate("PulsePatternWin", "list view"))
        self.tabWidget_periodic_pattern.setTabText(self.tabWidget_periodic_pattern.indexOf(self.tab_periodic_pattern), _translate("PulsePatternWin", "periodic pattern"))
        self.tabWidget_periodic_pattern.setTabText(self.tabWidget_periodic_pattern.indexOf(self.tab_simple), _translate("PulsePatternWin", "simple"))
        self.pushButton_reset_fpga.setText(_translate("PulsePatternWin", "reset fpga"))
        self.pushButton_stop.setText(_translate("PulsePatternWin", "stop pulse pattern"))
        self.pushButton_run_pattern.setText(_translate("PulsePatternWin", "run pulse pattern"))
        self.pushButton_close.setText(_translate("PulsePatternWin", "close and confirm"))
        self.label_2.setText(_translate("PulsePatternWin", "pulse generator state:"))
        self.label_ppg_state.setText(_translate("PulsePatternWin", "None"))
        self.menuHelp.setTitle(_translate("PulsePatternWin", "Help"))
        self.actionHelp.setText(_translate("PulsePatternWin", "help"))
        self.actionHelp.setShortcut(_translate("PulsePatternWin", "F1"))

