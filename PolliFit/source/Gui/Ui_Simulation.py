# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'E:\Users\Patrick\Documents\Python Projects\Tilda\PolliFit\source\Gui\Ui_Simulation.ui'
#
# Created by: PyQt5 UI code generator 5.13.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Simulation(object):
    def setupUi(self, Simulation):
        Simulation.setObjectName("Simulation")
        Simulation.resize(588, 402)
        self.horizontalLayout = QtWidgets.QHBoxLayout(Simulation)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(Simulation)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.listIsotopes = QtWidgets.QListWidget(Simulation)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.listIsotopes.sizePolicy().hasHeightForWidth())
        self.listIsotopes.setSizePolicy(sizePolicy)
        self.listIsotopes.setAlternatingRowColors(False)
        self.listIsotopes.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.listIsotopes.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.listIsotopes.setViewMode(QtWidgets.QListView.ListMode)
        self.listIsotopes.setModelColumn(0)
        self.listIsotopes.setObjectName("listIsotopes")
        item = QtWidgets.QListWidgetItem()
        self.listIsotopes.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listIsotopes.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listIsotopes.addItem(item)
        self.verticalLayout.addWidget(self.listIsotopes)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.dAccVolt = QtWidgets.QDoubleSpinBox(Simulation)
        self.dAccVolt.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.dAccVolt.setCorrectionMode(QtWidgets.QAbstractSpinBox.CorrectToNearestValue)
        self.dAccVolt.setDecimals(3)
        self.dAccVolt.setMinimum(-9999999.0)
        self.dAccVolt.setMaximum(9999999.0)
        self.dAccVolt.setSingleStep(0.01)
        self.dAccVolt.setObjectName("dAccVolt")
        self.gridLayout.addWidget(self.dAccVolt, 5, 1, 1, 1)
        self.coLineVar = QtWidgets.QComboBox(Simulation)
        self.coLineVar.setObjectName("coLineVar")
        self.coLineVar.addItem("")
        self.coLineVar.addItem("")
        self.gridLayout.addWidget(self.coLineVar, 0, 1, 1, 1)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.comboIso = QtWidgets.QComboBox(Simulation)
        self.comboIso.setObjectName("comboIso")
        self.comboIso.addItem("")
        self.comboIso.addItem("")
        self.comboIso.addItem("")
        self.horizontalLayout_5.addWidget(self.comboIso)
        self.label_7 = QtWidgets.QLabel(Simulation)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_5.addWidget(self.label_7)
        self.dIso = QtWidgets.QDoubleSpinBox(Simulation)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dIso.sizePolicy().hasHeightForWidth())
        self.dIso.setSizePolicy(sizePolicy)
        self.dIso.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.dIso.setCorrectionMode(QtWidgets.QAbstractSpinBox.CorrectToNearestValue)
        self.dIso.setPrefix("")
        self.dIso.setDecimals(3)
        self.dIso.setMinimum(-9999999.0)
        self.dIso.setMaximum(9999999.0)
        self.dIso.setSingleStep(0.001)
        self.dIso.setObjectName("dIso")
        self.horizontalLayout_5.addWidget(self.dIso)
        self.dIsoAmp = QtWidgets.QDoubleSpinBox(Simulation)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dIsoAmp.sizePolicy().hasHeightForWidth())
        self.dIsoAmp.setSizePolicy(sizePolicy)
        self.dIsoAmp.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.dIsoAmp.setCorrectionMode(QtWidgets.QAbstractSpinBox.CorrectToNearestValue)
        self.dIsoAmp.setDecimals(3)
        self.dIsoAmp.setMinimum(-9999999.0)
        self.dIsoAmp.setMaximum(9999999.0)
        self.dIsoAmp.setSingleStep(0.001)
        self.dIsoAmp.setObjectName("dIsoAmp")
        self.horizontalLayout_5.addWidget(self.dIsoAmp)
        self.gridLayout.addLayout(self.horizontalLayout_5, 7, 1, 1, 1)
        self.line_4 = QtWidgets.QFrame(Simulation)
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.gridLayout.addWidget(self.line_4, 8, 1, 1, 1)
        self.line_5 = QtWidgets.QFrame(Simulation)
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.gridLayout.addWidget(self.line_5, 6, 1, 1, 1)
        self.cAmplifier = QtWidgets.QCheckBox(Simulation)
        self.cAmplifier.setObjectName("cAmplifier")
        self.gridLayout.addWidget(self.cAmplifier, 4, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(Simulation)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 9, 0, 1, 1)
        self.dColFreq = QtWidgets.QDoubleSpinBox(Simulation)
        self.dColFreq.setStyleSheet("color: rgb(0, 0, 255);")
        self.dColFreq.setInputMethodHints(QtCore.Qt.ImhFormattedNumbersOnly)
        self.dColFreq.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.dColFreq.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.dColFreq.setCorrectionMode(QtWidgets.QAbstractSpinBox.CorrectToNearestValue)
        self.dColFreq.setDecimals(3)
        self.dColFreq.setMaximum(999999999999999.0)
        self.dColFreq.setSingleStep(0.001)
        self.dColFreq.setObjectName("dColFreq")
        self.gridLayout.addWidget(self.dColFreq, 1, 1, 1, 1)
        self.line_2 = QtWidgets.QFrame(Simulation)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.gridLayout.addWidget(self.line_2, 8, 0, 1, 1)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.dAmpSlope = QtWidgets.QDoubleSpinBox(Simulation)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dAmpSlope.sizePolicy().hasHeightForWidth())
        self.dAmpSlope.setSizePolicy(sizePolicy)
        self.dAmpSlope.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.dAmpSlope.setCorrectionMode(QtWidgets.QAbstractSpinBox.CorrectToNearestValue)
        self.dAmpSlope.setSuffix("")
        self.dAmpSlope.setDecimals(3)
        self.dAmpSlope.setMinimum(0.001)
        self.dAmpSlope.setMaximum(99999.0)
        self.dAmpSlope.setSingleStep(0.001)
        self.dAmpSlope.setProperty("value", 1.0)
        self.dAmpSlope.setObjectName("dAmpSlope")
        self.horizontalLayout_3.addWidget(self.dAmpSlope)
        self.label_4 = QtWidgets.QLabel(Simulation)
        self.label_4.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_3.addWidget(self.label_4)
        self.label_9 = QtWidgets.QLabel(Simulation)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_9.sizePolicy().hasHeightForWidth())
        self.label_9.setSizePolicy(sizePolicy)
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_3.addWidget(self.label_9)
        self.label_8 = QtWidgets.QLabel(Simulation)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_8.sizePolicy().hasHeightForWidth())
        self.label_8.setSizePolicy(sizePolicy)
        self.label_8.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_3.addWidget(self.label_8)
        self.dAmpOff = QtWidgets.QDoubleSpinBox(Simulation)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dAmpOff.sizePolicy().hasHeightForWidth())
        self.dAmpOff.setSizePolicy(sizePolicy)
        self.dAmpOff.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.dAmpOff.setCorrectionMode(QtWidgets.QAbstractSpinBox.CorrectToNearestValue)
        self.dAmpOff.setDecimals(3)
        self.dAmpOff.setMinimum(-9999999.0)
        self.dAmpOff.setMaximum(9999999.0)
        self.dAmpOff.setSingleStep(0.001)
        self.dAmpOff.setObjectName("dAmpOff")
        self.horizontalLayout_3.addWidget(self.dAmpOff)
        self.label_5 = QtWidgets.QLabel(Simulation)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_3.addWidget(self.label_5)
        self.gridLayout.addLayout(self.horizontalLayout_3, 4, 1, 1, 1)
        self.line_3 = QtWidgets.QFrame(Simulation)
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.gridLayout.addWidget(self.line_3, 6, 0, 1, 1)
        self.dAcolFreq = QtWidgets.QDoubleSpinBox(Simulation)
        self.dAcolFreq.setStyleSheet("color: rgb(255, 0, 0);")
        self.dAcolFreq.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.dAcolFreq.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.dAcolFreq.setCorrectionMode(QtWidgets.QAbstractSpinBox.CorrectToNearestValue)
        self.dAcolFreq.setDecimals(3)
        self.dAcolFreq.setMaximum(999999999999999.0)
        self.dAcolFreq.setSingleStep(0.001)
        self.dAcolFreq.setObjectName("dAcolFreq")
        self.gridLayout.addWidget(self.dAcolFreq, 2, 1, 1, 1)
        self.label_11 = QtWidgets.QLabel(Simulation)
        self.label_11.setObjectName("label_11")
        self.gridLayout.addWidget(self.label_11, 11, 0, 1, 1)
        self.label_6 = QtWidgets.QLabel(Simulation)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 5, 0, 1, 1)
        self.cIso = QtWidgets.QCheckBox(Simulation)
        self.cIso.setObjectName("cIso")
        self.gridLayout.addWidget(self.cIso, 7, 0, 1, 1)
        self.line_6 = QtWidgets.QFrame(Simulation)
        self.line_6.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_6.setObjectName("line_6")
        self.gridLayout.addWidget(self.line_6, 3, 1, 1, 1)
        self.label_10 = QtWidgets.QLabel(Simulation)
        self.label_10.setObjectName("label_10")
        self.gridLayout.addWidget(self.label_10, 10, 0, 1, 1)
        self.cColFreq = QtWidgets.QCheckBox(Simulation)
        self.cColFreq.setStyleSheet("color: rgb(0, 0, 255);")
        self.cColFreq.setObjectName("cColFreq")
        self.gridLayout.addWidget(self.cColFreq, 1, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(Simulation)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 0, 1, 1)
        self.line = QtWidgets.QFrame(Simulation)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.gridLayout.addWidget(self.line, 3, 0, 1, 1)
        self.cAcolFreq = QtWidgets.QCheckBox(Simulation)
        self.cAcolFreq.setStyleSheet("color: rgb(255, 0, 0);")
        self.cAcolFreq.setObjectName("cAcolFreq")
        self.gridLayout.addWidget(self.cAcolFreq, 2, 0, 1, 1)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.sCharge = QtWidgets.QSpinBox(Simulation)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sCharge.sizePolicy().hasHeightForWidth())
        self.sCharge.setSizePolicy(sizePolicy)
        self.sCharge.setMinimumSize(QtCore.QSize(50, 0))
        self.sCharge.setMinimum(-999)
        self.sCharge.setMaximum(999)
        self.sCharge.setProperty("value", 1)
        self.sCharge.setObjectName("sCharge")
        self.horizontalLayout_4.addWidget(self.sCharge)
        self.label_12 = QtWidgets.QLabel(Simulation)
        font = QtGui.QFont()
        font.setItalic(True)
        self.label_12.setFont(font)
        self.label_12.setStyleSheet("color: rgb(255, 0, 0);")
        self.label_12.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_12.setText("")
        self.label_12.setObjectName("label_12")
        self.horizontalLayout_4.addWidget(self.label_12)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem)
        self.gridLayout.addLayout(self.horizontalLayout_4, 11, 1, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.cInFreq = QtWidgets.QCheckBox(Simulation)
        self.cInFreq.setText("")
        self.cInFreq.setObjectName("cInFreq")
        self.horizontalLayout_2.addWidget(self.cInFreq)
        self.label_13 = QtWidgets.QLabel(Simulation)
        self.label_13.setObjectName("label_13")
        self.horizontalLayout_2.addWidget(self.label_13)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.gridLayout.addLayout(self.horizontalLayout_2, 9, 1, 1, 1)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.cNorm = QtWidgets.QCheckBox(Simulation)
        self.cNorm.setText("")
        self.cNorm.setChecked(True)
        self.cNorm.setObjectName("cNorm")
        self.horizontalLayout_6.addWidget(self.cNorm)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem2)
        self.gridLayout.addLayout(self.horizontalLayout_6, 10, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.pShow = QtWidgets.QPushButton(Simulation)
        self.pShow.setObjectName("pShow")
        self.verticalLayout.addWidget(self.pShow)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem3)
        self.horizontalLayout.addLayout(self.verticalLayout)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem4)

        self.retranslateUi(Simulation)
        QtCore.QMetaObject.connectSlotsByName(Simulation)

    def retranslateUi(self, Simulation):
        _translate = QtCore.QCoreApplication.translate
        Simulation.setWindowTitle(_translate("Simulation", "Form"))
        self.label.setText(_translate("Simulation", "Collinear Spectrum Simulation"))
        __sortingEnabled = self.listIsotopes.isSortingEnabled()
        self.listIsotopes.setSortingEnabled(False)
        item = self.listIsotopes.item(0)
        item.setText(_translate("Simulation", "test1"))
        item = self.listIsotopes.item(1)
        item.setText(_translate("Simulation", "test2"))
        item = self.listIsotopes.item(2)
        item.setText(_translate("Simulation", "test3"))
        self.listIsotopes.setSortingEnabled(__sortingEnabled)
        self.dAccVolt.setSuffix(_translate("Simulation", " V"))
        self.coLineVar.setItemText(0, _translate("Simulation", "b1"))
        self.coLineVar.setItemText(1, _translate("Simulation", "b2"))
        self.comboIso.setItemText(0, _translate("Simulation", "test1"))
        self.comboIso.setItemText(1, _translate("Simulation", "test2"))
        self.comboIso.setItemText(2, _translate("Simulation", "test3"))
        self.label_7.setText(_translate("Simulation", "@"))
        self.dIso.setSuffix(_translate("Simulation", " V"))
        self.dIsoAmp.setSuffix(_translate("Simulation", " V(Amp)"))
        self.cAmplifier.setText(_translate("Simulation", "Amplifier"))
        self.label_3.setText(_translate("Simulation", "In frequency?"))
        self.dColFreq.setSuffix(_translate("Simulation", " MHz"))
        self.label_4.setText(_translate("Simulation", "<nobr>&times; <i>U</i></nobr>"))
        self.label_9.setText(_translate("Simulation", "+"))
        self.label_8.setText(_translate("Simulation", "("))
        self.dAmpOff.setSuffix(_translate("Simulation", " V"))
        self.label_5.setText(_translate("Simulation", ")"))
        self.dAcolFreq.setSuffix(_translate("Simulation", " MHz"))
        self.label_11.setText(_translate("Simulation", "Scan charge state:"))
        self.label_6.setText(_translate("Simulation", "      AccVolt"))
        self.cIso.setText(_translate("Simulation", "Isotope"))
        self.label_10.setText(_translate("Simulation", "Equal intensities?"))
        self.cColFreq.setText(_translate("Simulation", "Col. laser freq."))
        self.label_2.setText(_translate("Simulation", "Lineshape:"))
        self.cAcolFreq.setText(_translate("Simulation", "Acol. laser freq."))
        self.label_13.setText(_translate("Simulation", "[eV]"))
        self.pShow.setText(_translate("Simulation", "Show spectrum"))
