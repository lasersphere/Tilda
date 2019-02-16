# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_JobStacker.ui'
#
# Created by: PyQt5 UI code generator 5.5
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_JobStacker(object):
    def setupUi(self, JobStacker):
        JobStacker.setObjectName("JobStacker")
        JobStacker.resize(407, 370)
        self.centralwidget = QtWidgets.QWidget(JobStacker)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.list_joblist = QtWidgets.QListWidget(self.centralwidget)
        self.list_joblist.setDragDropMode(QtWidgets.QAbstractItemView.DragDrop)
        self.list_joblist.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.list_joblist.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.list_joblist.setObjectName("list_joblist")
        item = QtWidgets.QListWidgetItem()
        self.list_joblist.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.list_joblist.addItem(item)
        self.horizontalLayout.addWidget(self.list_joblist)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setContentsMargins(-1, -1, 0, -1)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.pb_add = QtWidgets.QPushButton(self.centralwidget)
        self.pb_add.setObjectName("pb_add")
        self.verticalLayout_2.addWidget(self.pb_add)
        self.pb_del = QtWidgets.QPushButton(self.centralwidget)
        self.pb_del.setObjectName("pb_del")
        self.verticalLayout_2.addWidget(self.pb_del)
        self.pb_repetitions = QtWidgets.QPushButton(self.centralwidget)
        self.pb_repetitions.setObjectName("pb_repetitions")
        self.verticalLayout_2.addWidget(self.pb_repetitions)
        self.pb_save = QtWidgets.QPushButton(self.centralwidget)
        self.pb_save.setObjectName("pb_save")
        self.verticalLayout_2.addWidget(self.pb_save)
        self.pb_load = QtWidgets.QPushButton(self.centralwidget)
        self.pb_load.setObjectName("pb_load")
        self.verticalLayout_2.addWidget(self.pb_load)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.pb_run = QtWidgets.QPushButton(self.centralwidget)
        self.pb_run.setObjectName("pb_run")
        self.verticalLayout.addWidget(self.pb_run)
        JobStacker.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(JobStacker)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 407, 21))
        self.menubar.setObjectName("menubar")
        JobStacker.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(JobStacker)
        self.statusbar.setObjectName("statusbar")
        JobStacker.setStatusBar(self.statusbar)

        self.retranslateUi(JobStacker)
        QtCore.QMetaObject.connectSlotsByName(JobStacker)

    def retranslateUi(self, JobStacker):
        _translate = QtCore.QCoreApplication.translate
        JobStacker.setWindowTitle(_translate("JobStacker", "Job Stacker"))
        __sortingEnabled = self.list_joblist.isSortingEnabled()
        self.list_joblist.setSortingEnabled(False)
        item = self.list_joblist.item(0)
        item.setText(_translate("JobStacker", "iso_name | seq_type | #reps_as_go | #reps_new_file"))
        item = self.list_joblist.item(1)
        item.setText(_translate("JobStacker", "TestDummy | trsdummy | 2 | 3"))
        self.list_joblist.setSortingEnabled(__sortingEnabled)
        self.pb_add.setText(_translate("JobStacker", "Add Job"))
        self.pb_del.setText(_translate("JobStacker", "Del Selected"))
        self.pb_repetitions.setText(_translate("JobStacker", "Change Reps"))
        self.pb_save.setText(_translate("JobStacker", "Save .txt"))
        self.pb_load.setText(_translate("JobStacker", "Load .txt"))
        self.pb_run.setText(_translate("JobStacker", "Run"))

