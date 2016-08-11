# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_Crawler.ui'
#
# Created: Thu Aug  4 20:40:00 2016
#      by: PyQt5 UI code generator 5.3.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Crawler(object):
    def setupUi(self, Crawler):
        Crawler.setObjectName("Crawler")
        Crawler.resize(526, 357)
        self.verticalLayout = QtWidgets.QVBoxLayout(Crawler)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(Crawler)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setMinimumSize(QtCore.QSize(80, 0))
        self.label.setMaximumSize(QtCore.QSize(80, 16777215))
        self.label.setWordWrap(True)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.path = QtWidgets.QLineEdit(Crawler)
        self.path.setObjectName("path")
        self.horizontalLayout.addWidget(self.path)
        self.recursive = QtWidgets.QCheckBox(Crawler)
        self.recursive.setChecked(True)
        self.recursive.setObjectName("recursive")
        self.horizontalLayout.addWidget(self.recursive)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.bcrawl = QtWidgets.QPushButton(Crawler)
        self.bcrawl.setObjectName("bcrawl")
        self.verticalLayout.addWidget(self.bcrawl)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.formLayout.setObjectName("formLayout")
        self.label_2 = QtWidgets.QLabel(Crawler)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.lineEdit_sql_cmd = QtWidgets.QLineEdit(Crawler)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_sql_cmd.sizePolicy().hasHeightForWidth())
        self.lineEdit_sql_cmd.setSizePolicy(sizePolicy)
        self.lineEdit_sql_cmd.setMinimumSize(QtCore.QSize(100, 0))
        self.lineEdit_sql_cmd.setObjectName("lineEdit_sql_cmd")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit_sql_cmd)
        self.pushButton_save_sql = QtWidgets.QPushButton(Crawler)
        self.pushButton_save_sql.setObjectName("pushButton_save_sql")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.pushButton_save_sql)
        self.pushButton_load_sql = QtWidgets.QPushButton(Crawler)
        self.pushButton_load_sql.setObjectName("pushButton_load_sql")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.pushButton_load_sql)
        self.verticalLayout.addLayout(self.formLayout)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)

        self.retranslateUi(Crawler)
        QtCore.QMetaObject.connectSlotsByName(Crawler)

    def retranslateUi(self, Crawler):
        _translate = QtCore.QCoreApplication.translate
        Crawler.setWindowTitle(_translate("Crawler", "Form"))
        self.label.setText(_translate("Crawler", "Data Folder (relative to DB)"))
        self.recursive.setText(_translate("Crawler", "Recursive"))
        self.bcrawl.setText(_translate("Crawler", "Crawl"))
        self.label_2.setText(_translate("Crawler", "sql command after crawl"))
        self.pushButton_save_sql.setText(_translate("Crawler", "save sql cmd"))
        self.pushButton_load_sql.setText(_translate("Crawler", "load sql cmd"))

