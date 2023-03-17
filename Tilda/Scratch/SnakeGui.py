"""
Created on 

@author: simkaufm

Module Description:
"""

# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt5 UI code generator 5.7
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Field(object):
    def setupUi(self, Field):
        Field.setObjectName("Field")
        Field.resize(270, 570)
        Field.setMinimumSize(QtCore.QSize(270, 570))
        Field.setMaximumSize(QtCore.QSize(270, 570))
        self.QGraph_field = QtWidgets.QGraphicsView(Field)
        self.QGraph_field.setGeometry(QtCore.QRect(0, 300, 270, 270))
        self.QGraph_field.setMinimumSize(QtCore.QSize(270, 270))
        self.QGraph_field.setMaximumSize(QtCore.QSize(270, 270))
        self.QGraph_field.setFocusPolicy(QtCore.Qt.NoFocus)
        self.QGraph_field.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.QGraph_field.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.QGraph_field.setSceneRect(QtCore.QRectF(0.0, 0.0, 270.0, 270.0))
        self.QGraph_field.setObjectName("QGraph_field")
        self.verticalLayoutWidget = QtWidgets.QWidget(Field)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 10, 251, 271))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.vertical_layout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.vertical_layout.setContentsMargins(0, 0, 0, 0)
        self.vertical_layout.setObjectName("vertical_layout")
        self.QText_status = QtWidgets.QTextBrowser(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.QText_status.setFont(font)
        self.QText_status.setFocusPolicy(QtCore.Qt.NoFocus)
        self.QText_status.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.QText_status.setReadOnly(True)
        self.QText_status.setTabStopWidth(79)
        self.QText_status.setObjectName("QText_status")
        self.vertical_layout.addWidget(self.QText_status)
        self.horizontal_layout1 = QtWidgets.QHBoxLayout()
        self.horizontal_layout1.setObjectName("horizontal_layout1")
        self.QBut_start = QtWidgets.QPushButton(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.QBut_start.setFont(font)
        self.QBut_start.setFocusPolicy(QtCore.Qt.NoFocus)
        self.QBut_start.setObjectName("QBut_start")
        self.horizontal_layout1.addWidget(self.QBut_start)
        self.QBut_rules = QtWidgets.QPushButton(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.QBut_rules.setFont(font)
        self.QBut_rules.setFocusPolicy(QtCore.Qt.NoFocus)
        self.QBut_rules.setObjectName("QBut_rules")
        self.horizontal_layout1.addWidget(self.QBut_rules)
        self.vertical_layout.addLayout(self.horizontal_layout1)
        self.horizontal_layout2 = QtWidgets.QHBoxLayout()
        self.horizontal_layout2.setObjectName("horizontal_layout2")
        self.QBut_reset = QtWidgets.QPushButton(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.QBut_reset.setFont(font)
        self.QBut_reset.setFocusPolicy(QtCore.Qt.NoFocus)
        self.QBut_reset.setObjectName("QBut_reset")
        self.horizontal_layout2.addWidget(self.QBut_reset)
        self.QBut_main = QtWidgets.QPushButton(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.QBut_main.setFont(font)
        self.QBut_main.setFocusPolicy(QtCore.Qt.NoFocus)
        self.QBut_main.setObjectName("QBut_main")
        self.horizontal_layout2.addWidget(self.QBut_main)
        self.vertical_layout.addLayout(self.horizontal_layout2)

        self.retranslateUi(Field)
        QtCore.QMetaObject.connectSlotsByName(Field)

    def retranslateUi(self, Field):
        _translate = QtCore.QCoreApplication.translate
        Field.setWindowTitle(_translate("Field", "PainterField"))
        self.QBut_start.setText(_translate("Field", "Start"))
        self.QBut_rules.setText(_translate("Field", "Rules"))
        self.QBut_reset.setText(_translate("Field", "Reset"))
        self.QBut_main.setText(_translate("Field", "Main menu"))