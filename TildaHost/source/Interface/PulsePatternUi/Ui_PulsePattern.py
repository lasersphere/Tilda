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
        PulsePatternWin.resize(346, 358)
        self.centralwidget = QtWidgets.QWidget(PulsePatternWin)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.tab_1_graph_view = QtWidgets.QWidget()
        self.tab_1_graph_view.setObjectName("tab_1_graph_view")
        self.tabWidget.addTab(self.tab_1_graph_view, "")
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
        self.tabWidget.addTab(self.tab_list_view, "")
        self.verticalLayout.addWidget(self.tabWidget)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
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
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_3.addWidget(self.label_2)
        self.label_ppg_state = QtWidgets.QLabel(self.centralwidget)
        self.label_ppg_state.setObjectName("label_ppg_state")
        self.horizontalLayout_3.addWidget(self.label_ppg_state)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        PulsePatternWin.setCentralWidget(self.centralwidget)

        self.retranslateUi(PulsePatternWin)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(PulsePatternWin)

    def retranslateUi(self, PulsePatternWin):
        _translate = QtCore.QCoreApplication.translate
        PulsePatternWin.setWindowTitle(_translate("PulsePatternWin", "MainWindow"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_1_graph_view),
                                  _translate("PulsePatternWin", "graph view"))
        self.pushButton_add_cmd.setText(_translate("PulsePatternWin", "add cmd"))
        self.pushButton_remove_selected.setText(_translate("PulsePatternWin", "rem sel."))
        self.pushButton_load_txt.setText(_translate("PulsePatternWin", "load .txt"))
        self.pushButton_save_txt.setText(_translate("PulsePatternWin", "save .txt"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_list_view),
                                  _translate("PulsePatternWin", "list view"))
        self.pushButton_stop.setText(_translate("PulsePatternWin", "stop pulse pattern"))
        self.pushButton_run_pattern.setText(_translate("PulsePatternWin", "run pulse pattern"))
        self.pushButton_close.setText(_translate("PulsePatternWin", "close and confirm"))
        self.label_2.setText(_translate("PulsePatternWin", "pulse generator state:"))
        self.label_ppg_state.setText(_translate("PulsePatternWin", "None"))
