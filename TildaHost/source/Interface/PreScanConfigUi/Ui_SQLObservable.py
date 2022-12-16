# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'E:\Users\Patrick\Documents\Python Projects\Tilda\TildaHost\source\Interface\PreScanConfigUi\Ui_SQLObservable.ui'
#
# Created by: PyQt5 UI code generator 5.13.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_SQLObservable(object):
    def setupUi(self, SQLObservable):
        SQLObservable.setObjectName("SQLObservable")
        SQLObservable.resize(398, 60)
        self.gridLayout = QtWidgets.QGridLayout(SQLObservable)
        self.gridLayout.setObjectName("gridLayout")
        self.label_channel = QtWidgets.QLabel(SQLObservable)
        self.label_channel.setObjectName("label_channel")
        self.gridLayout.addWidget(self.label_channel, 0, 1, 1, 1)
        self.label_aggregation = QtWidgets.QLabel(SQLObservable)
        self.label_aggregation.setObjectName("label_aggregation")
        self.gridLayout.addWidget(self.label_aggregation, 0, 2, 1, 1)
        self.combo_channel = QtWidgets.QComboBox(SQLObservable)
        self.combo_channel.setObjectName("combo_channel")
        self.gridLayout.addWidget(self.combo_channel, 1, 1, 1, 1)
        self.b_delete = QtWidgets.QPushButton(SQLObservable)
        self.b_delete.setMaximumSize(QtCore.QSize(50, 16777215))
        self.b_delete.setObjectName("b_delete")
        self.gridLayout.addWidget(self.b_delete, 1, 3, 1, 1)
        self.combo_table = QtWidgets.QComboBox(SQLObservable)
        self.combo_table.setObjectName("combo_table")
        self.gridLayout.addWidget(self.combo_table, 1, 0, 1, 1)
        self.combo_aggregation = QtWidgets.QComboBox(SQLObservable)
        self.combo_aggregation.setObjectName("combo_aggregation")
        self.gridLayout.addWidget(self.combo_aggregation, 1, 2, 1, 1)
        self.label_table = QtWidgets.QLabel(SQLObservable)
        self.label_table.setObjectName("label_table")
        self.gridLayout.addWidget(self.label_table, 0, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 1, 4, 1, 1)
        self.gridLayout.setColumnStretch(4, 1)
        self.gridLayout.setRowStretch(0, 1)
        self.gridLayout.setRowStretch(1, 1)

        self.retranslateUi(SQLObservable)
        QtCore.QMetaObject.connectSlotsByName(SQLObservable)

    def retranslateUi(self, SQLObservable):
        _translate = QtCore.QCoreApplication.translate
        SQLObservable.setWindowTitle(_translate("SQLObservable", "Form"))
        self.label_channel.setText(_translate("SQLObservable", "Channel"))
        self.label_aggregation.setText(_translate("SQLObservable", "Aggregation <i>- optional</i>"))
        self.b_delete.setText(_translate("SQLObservable", "X"))
        self.label_table.setText(_translate("SQLObservable", "Table"))
