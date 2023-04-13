# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_KingFitter.ui'
#
# Created by: PyQt5 UI code generator 5.5
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_KingFitter(object):
    def setupUi(self, KingFitter):
        KingFitter.setObjectName("KingFitter")
        KingFitter.resize(313, 210)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(KingFitter)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.bKingFit = QtWidgets.QPushButton(KingFitter)
        self.bKingFit.setObjectName("bKingFit")
        self.horizontalLayout_2.addWidget(self.bKingFit)
        self.alphaTrue = QtWidgets.QCheckBox(KingFitter)
        self.alphaTrue.setEnabled(True)
        self.alphaTrue.setChecked(False)
        self.alphaTrue.setObjectName("alphaTrue")
        self.horizontalLayout_2.addWidget(self.alphaTrue)
        self.sAlpha = QtWidgets.QSpinBox(KingFitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sAlpha.sizePolicy().hasHeightForWidth())
        self.sAlpha.setSizePolicy(sizePolicy)
        self.sAlpha.setMinimum(-2000)
        self.sAlpha.setMaximum(2000)
        self.sAlpha.setProperty("value", 0)
        self.sAlpha.setObjectName("sAlpha")
        self.horizontalLayout_2.addWidget(self.sAlpha)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.runSelect = QtWidgets.QComboBox(KingFitter)
        self.runSelect.setObjectName("runSelect")
        self.horizontalLayout.addWidget(self.runSelect)
        self.allRuns = QtWidgets.QCheckBox(KingFitter)
        self.allRuns.setChecked(True)
        self.allRuns.setObjectName("allRuns")
        self.horizontalLayout.addWidget(self.allRuns)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.isoList = QtWidgets.QListWidget(KingFitter)
        self.isoList.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.isoList.setObjectName("isoList")
        self.verticalLayout.addWidget(self.isoList)
        self.pushButton_select_all = QtWidgets.QPushButton(KingFitter)
        self.pushButton_select_all.setObjectName("pushButton_select_all")
        self.verticalLayout.addWidget(self.pushButton_select_all)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.bCalcChargeRadii = QtWidgets.QPushButton(KingFitter)
        self.bCalcChargeRadii.setObjectName("bCalcChargeRadii")
        self.verticalLayout_2.addWidget(self.bCalcChargeRadii)

        self.retranslateUi(KingFitter)
        QtCore.QMetaObject.connectSlotsByName(KingFitter)

    def retranslateUi(self, KingFitter):
        _translate = QtCore.QCoreApplication.translate
        KingFitter.setWindowTitle(_translate("KingFitter", "Form"))
        self.bKingFit.setText(_translate("KingFitter", "perform King fit!"))
        self.alphaTrue.setText(_translate("KingFitter", "find best alpha? alpha = "))
        self.allRuns.setText(_translate("KingFitter", "All runs"))
        self.pushButton_select_all.setText(_translate("KingFitter", "select/deselect All"))
        self.bCalcChargeRadii.setText(_translate("KingFitter", "calculate charge radii"))

