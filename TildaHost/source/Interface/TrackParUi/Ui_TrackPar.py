# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_TrackPar.ui'
#
# Created by: PyQt5 UI code generator 5.5
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindowTrackPars(object):
    def setupUi(self, MainWindowTrackPars):
        MainWindowTrackPars.setObjectName("MainWindowTrackPars")
        MainWindowTrackPars.resize(428, 527)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("track_icon.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindowTrackPars.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindowTrackPars)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.specificSequencerSettings = QtWidgets.QWidget(self.centralwidget)
        self.specificSequencerSettings.setObjectName("specificSequencerSettings")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.specificSequencerSettings)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout.addWidget(self.specificSequencerSettings)
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout.addWidget(self.line)
        self.commonSequencerSettings = QtWidgets.QWidget(self.centralwidget)
        self.commonSequencerSettings.setObjectName("commonSequencerSettings")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.commonSequencerSettings)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.tabwidget = QtWidgets.QTabWidget(self.commonSequencerSettings)
        self.tabwidget.setObjectName("tabwidget")
        self.seqComSet = QtWidgets.QWidget()
        self.seqComSet.setObjectName("seqComSet")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.seqComSet)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.lineEdit_activePmtList = QtWidgets.QLineEdit(self.seqComSet)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_activePmtList.sizePolicy().hasHeightForWidth())
        self.lineEdit_activePmtList.setSizePolicy(sizePolicy)
        self.lineEdit_activePmtList.setText("")
        self.lineEdit_activePmtList.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.lineEdit_activePmtList.setObjectName("lineEdit_activePmtList")
        self.gridLayout_4.addWidget(self.lineEdit_activePmtList, 10, 1, 1, 1)
        self.label_nOfScans = QtWidgets.QLabel(self.seqComSet)
        self.label_nOfScans.setObjectName("label_nOfScans")
        self.gridLayout_4.addWidget(self.label_nOfScans, 5, 0, 1, 1)
        self.doubleSpinBox_postAccOffsetVolt = QtWidgets.QDoubleSpinBox(self.seqComSet)
        self.doubleSpinBox_postAccOffsetVolt.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.doubleSpinBox_postAccOffsetVolt.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.doubleSpinBox_postAccOffsetVolt.setKeyboardTracking(False)
        self.doubleSpinBox_postAccOffsetVolt.setMaximum(10000.0)
        self.doubleSpinBox_postAccOffsetVolt.setObjectName("doubleSpinBox_postAccOffsetVolt")
        self.gridLayout_4.addWidget(self.doubleSpinBox_postAccOffsetVolt, 9, 1, 1, 1)
        self.label_invertScan = QtWidgets.QLabel(self.seqComSet)
        self.label_invertScan.setObjectName("label_invertScan")
        self.gridLayout_4.addWidget(self.label_invertScan, 6, 0, 1, 1)
        self.label_dacStartV_set = QtWidgets.QLabel(self.seqComSet)
        self.label_dacStartV_set.setObjectName("label_dacStartV_set")
        self.gridLayout_4.addWidget(self.label_dacStartV_set, 1, 2, 1, 1)
        self.doubleSpinBox_dacStartV = QtWidgets.QDoubleSpinBox(self.seqComSet)
        self.doubleSpinBox_dacStartV.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.doubleSpinBox_dacStartV.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.doubleSpinBox_dacStartV.setKeyboardTracking(False)
        self.doubleSpinBox_dacStartV.setDecimals(8)
        self.doubleSpinBox_dacStartV.setMinimum(-10.0)
        self.doubleSpinBox_dacStartV.setMaximum(10.0)
        self.doubleSpinBox_dacStartV.setSingleStep(0.01)
        self.doubleSpinBox_dacStartV.setObjectName("doubleSpinBox_dacStartV")
        self.gridLayout_4.addWidget(self.doubleSpinBox_dacStartV, 1, 1, 1, 1)
        self.checkBox_invertScan = QtWidgets.QCheckBox(self.seqComSet)
        self.checkBox_invertScan.setText("")
        self.checkBox_invertScan.setObjectName("checkBox_invertScan")
        self.gridLayout_4.addWidget(self.checkBox_invertScan, 6, 1, 1, 1)
        self.label_activePmtList = QtWidgets.QLabel(self.seqComSet)
        self.label_activePmtList.setObjectName("label_activePmtList")
        self.gridLayout_4.addWidget(self.label_activePmtList, 10, 0, 1, 1)
        self.label_postAccOffsetVolt = QtWidgets.QLabel(self.seqComSet)
        self.label_postAccOffsetVolt.setObjectName("label_postAccOffsetVolt")
        self.gridLayout_4.addWidget(self.label_postAccOffsetVolt, 9, 0, 1, 1)
        self.label_postAccOffsetVolt_set = QtWidgets.QLabel(self.seqComSet)
        self.label_postAccOffsetVolt_set.setObjectName("label_postAccOffsetVolt_set")
        self.gridLayout_4.addWidget(self.label_postAccOffsetVolt_set, 9, 2, 1, 1)
        self.spinBox_nOfScans = QtWidgets.QSpinBox(self.seqComSet)
        self.spinBox_nOfScans.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.spinBox_nOfScans.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.spinBox_nOfScans.setKeyboardTracking(False)
        self.spinBox_nOfScans.setMaximum(10000000)
        self.spinBox_nOfScans.setObjectName("spinBox_nOfScans")
        self.gridLayout_4.addWidget(self.spinBox_nOfScans, 5, 1, 1, 1)
        self.label_nOfScans_set = QtWidgets.QLabel(self.seqComSet)
        self.label_nOfScans_set.setObjectName("label_nOfScans_set")
        self.gridLayout_4.addWidget(self.label_nOfScans_set, 5, 2, 1, 1)
        self.checkBox_colDirTrue = QtWidgets.QCheckBox(self.seqComSet)
        self.checkBox_colDirTrue.setText("")
        self.checkBox_colDirTrue.setObjectName("checkBox_colDirTrue")
        self.gridLayout_4.addWidget(self.checkBox_colDirTrue, 11, 1, 1, 1)
        self.label_colDirTrue = QtWidgets.QLabel(self.seqComSet)
        self.label_colDirTrue.setObjectName("label_colDirTrue")
        self.gridLayout_4.addWidget(self.label_colDirTrue, 11, 0, 1, 1)
        self.label_colDirTrue_set = QtWidgets.QLabel(self.seqComSet)
        self.label_colDirTrue_set.setObjectName("label_colDirTrue_set")
        self.gridLayout_4.addWidget(self.label_colDirTrue_set, 11, 2, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.seqComSet)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy)
        self.label_5.setObjectName("label_5")
        self.gridLayout_4.addWidget(self.label_5, 0, 2, 1, 1)
        self.label_dacStartV = QtWidgets.QLabel(self.seqComSet)
        self.label_dacStartV.setObjectName("label_dacStartV")
        self.gridLayout_4.addWidget(self.label_dacStartV, 1, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.seqComSet)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy)
        self.label_4.setObjectName("label_4")
        self.gridLayout_4.addWidget(self.label_4, 0, 1, 1, 1)
        self.label_postAccOffsetVoltControl = QtWidgets.QLabel(self.seqComSet)
        self.label_postAccOffsetVoltControl.setObjectName("label_postAccOffsetVoltControl")
        self.gridLayout_4.addWidget(self.label_postAccOffsetVoltControl, 7, 0, 1, 1)
        self.label_dacStopV_set = QtWidgets.QLabel(self.seqComSet)
        self.label_dacStopV_set.setObjectName("label_dacStopV_set")
        self.gridLayout_4.addWidget(self.label_dacStopV_set, 2, 2, 1, 1)
        self.label_dacStepSizeV = QtWidgets.QLabel(self.seqComSet)
        self.label_dacStepSizeV.setObjectName("label_dacStepSizeV")
        self.gridLayout_4.addWidget(self.label_dacStepSizeV, 3, 0, 1, 1)
        self.label_dacStopV = QtWidgets.QLabel(self.seqComSet)
        self.label_dacStopV.setObjectName("label_dacStopV")
        self.gridLayout_4.addWidget(self.label_dacStopV, 2, 0, 1, 1)
        self.label_nOfSteps_set = QtWidgets.QLabel(self.seqComSet)
        self.label_nOfSteps_set.setObjectName("label_nOfSteps_set")
        self.gridLayout_4.addWidget(self.label_nOfSteps_set, 4, 2, 1, 1)
        self.spinBox_nOfSteps = QtWidgets.QSpinBox(self.seqComSet)
        self.spinBox_nOfSteps.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.spinBox_nOfSteps.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.spinBox_nOfSteps.setKeyboardTracking(False)
        self.spinBox_nOfSteps.setMaximum(100000000)
        self.spinBox_nOfSteps.setObjectName("spinBox_nOfSteps")
        self.gridLayout_4.addWidget(self.spinBox_nOfSteps, 4, 1, 1, 1)
        self.label_dacStepSizeV_set = QtWidgets.QLabel(self.seqComSet)
        self.label_dacStepSizeV_set.setObjectName("label_dacStepSizeV_set")
        self.gridLayout_4.addWidget(self.label_dacStepSizeV_set, 3, 2, 1, 1)
        self.label_nOfSteps = QtWidgets.QLabel(self.seqComSet)
        self.label_nOfSteps.setObjectName("label_nOfSteps")
        self.gridLayout_4.addWidget(self.label_nOfSteps, 4, 0, 1, 1)
        self.label_activePmtList_set = QtWidgets.QLabel(self.seqComSet)
        self.label_activePmtList_set.setObjectName("label_activePmtList_set")
        self.gridLayout_4.addWidget(self.label_activePmtList_set, 10, 2, 1, 1)
        self.doubleSpinBox_dacStopV = QtWidgets.QDoubleSpinBox(self.seqComSet)
        self.doubleSpinBox_dacStopV.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.doubleSpinBox_dacStopV.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.doubleSpinBox_dacStopV.setKeyboardTracking(False)
        self.doubleSpinBox_dacStopV.setDecimals(8)
        self.doubleSpinBox_dacStopV.setMinimum(-10.0)
        self.doubleSpinBox_dacStopV.setMaximum(10.0)
        self.doubleSpinBox_dacStopV.setSingleStep(0.01)
        self.doubleSpinBox_dacStopV.setObjectName("doubleSpinBox_dacStopV")
        self.gridLayout_4.addWidget(self.doubleSpinBox_dacStopV, 2, 1, 1, 1)
        self.label_invertScan_set = QtWidgets.QLabel(self.seqComSet)
        self.label_invertScan_set.setObjectName("label_invertScan_set")
        self.gridLayout_4.addWidget(self.label_invertScan_set, 6, 2, 1, 1)
        self.doubleSpinBox_dacStepSizeV = QtWidgets.QDoubleSpinBox(self.seqComSet)
        self.doubleSpinBox_dacStepSizeV.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.doubleSpinBox_dacStepSizeV.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.doubleSpinBox_dacStepSizeV.setKeyboardTracking(False)
        self.doubleSpinBox_dacStepSizeV.setDecimals(8)
        self.doubleSpinBox_dacStepSizeV.setMinimum(-10.0)
        self.doubleSpinBox_dacStepSizeV.setMaximum(10.0)
        self.doubleSpinBox_dacStepSizeV.setSingleStep(0.01)
        self.doubleSpinBox_dacStepSizeV.setObjectName("doubleSpinBox_dacStepSizeV")
        self.gridLayout_4.addWidget(self.doubleSpinBox_dacStepSizeV, 3, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.seqComSet)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        self.label_3.setObjectName("label_3")
        self.gridLayout_4.addWidget(self.label_3, 0, 0, 1, 1)
        self.label_kepco_start = QtWidgets.QLabel(self.seqComSet)
        self.label_kepco_start.setObjectName("label_kepco_start")
        self.gridLayout_4.addWidget(self.label_kepco_start, 1, 3, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.seqComSet)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_7.sizePolicy().hasHeightForWidth())
        self.label_7.setSizePolicy(sizePolicy)
        self.label_7.setObjectName("label_7")
        self.gridLayout_4.addWidget(self.label_7, 0, 3, 1, 1)
        self.label_kepco_stop = QtWidgets.QLabel(self.seqComSet)
        self.label_kepco_stop.setObjectName("label_kepco_stop")
        self.gridLayout_4.addWidget(self.label_kepco_stop, 2, 3, 1, 1)
        self.label_kepco_step = QtWidgets.QLabel(self.seqComSet)
        self.label_kepco_step.setObjectName("label_kepco_step")
        self.gridLayout_4.addWidget(self.label_kepco_step, 3, 3, 1, 1)
        self.comboBox_postAccOffsetVoltControl = QtWidgets.QComboBox(self.seqComSet)
        self.comboBox_postAccOffsetVoltControl.setObjectName("comboBox_postAccOffsetVoltControl")
        self.comboBox_postAccOffsetVoltControl.addItem("")
        self.comboBox_postAccOffsetVoltControl.addItem("")
        self.comboBox_postAccOffsetVoltControl.addItem("")
        self.comboBox_postAccOffsetVoltControl.addItem("")
        self.gridLayout_4.addWidget(self.comboBox_postAccOffsetVoltControl, 7, 1, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 0, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        self.gridLayout_4.addItem(spacerItem, 12, 1, 1, 1)
        self.pushButton_postAccOffsetVolt = QtWidgets.QPushButton(self.seqComSet)
        self.pushButton_postAccOffsetVolt.setObjectName("pushButton_postAccOffsetVolt")
        self.gridLayout_4.addWidget(self.pushButton_postAccOffsetVolt, 9, 3, 1, 1)
        self.label_postAccOffsetVoltControl_set = QtWidgets.QLabel(self.seqComSet)
        self.label_postAccOffsetVoltControl_set.setObjectName("label_postAccOffsetVoltControl_set")
        self.gridLayout_4.addWidget(self.label_postAccOffsetVoltControl_set, 7, 3, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.seqComSet)
        self.label_11.setObjectName("label_11")
        self.gridLayout_4.addWidget(self.label_11, 7, 2, 1, 1)
        self.tabwidget.addTab(self.seqComSet, "")
        self.advancedSeqSettings = QtWidgets.QWidget()
        self.advancedSeqSettings.setObjectName("advancedSeqSettings")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.advancedSeqSettings)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.doubleSpinBox_waitForKepco_muS = QtWidgets.QDoubleSpinBox(self.advancedSeqSettings)
        self.doubleSpinBox_waitForKepco_muS.setFrame(True)
        self.doubleSpinBox_waitForKepco_muS.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.doubleSpinBox_waitForKepco_muS.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.doubleSpinBox_waitForKepco_muS.setKeyboardTracking(False)
        self.doubleSpinBox_waitForKepco_muS.setDecimals(3)
        self.doubleSpinBox_waitForKepco_muS.setMaximum(10000.0)
        self.doubleSpinBox_waitForKepco_muS.setSingleStep(0.5)
        self.doubleSpinBox_waitForKepco_muS.setObjectName("doubleSpinBox_waitForKepco_muS")
        self.gridLayout_5.addWidget(self.doubleSpinBox_waitForKepco_muS, 1, 1, 1, 1)
        self.label_waitForKepco_muS_set = QtWidgets.QLabel(self.advancedSeqSettings)
        self.label_waitForKepco_muS_set.setObjectName("label_waitForKepco_muS_set")
        self.gridLayout_5.addWidget(self.label_waitForKepco_muS_set, 1, 2, 1, 1)
        self.label_waitForKepco_muS = QtWidgets.QLabel(self.advancedSeqSettings)
        self.label_waitForKepco_muS.setObjectName("label_waitForKepco_muS")
        self.gridLayout_5.addWidget(self.label_waitForKepco_muS, 1, 0, 1, 1)
        self.label_waitAfterReset_muS = QtWidgets.QLabel(self.advancedSeqSettings)
        self.label_waitAfterReset_muS.setObjectName("label_waitAfterReset_muS")
        self.gridLayout_5.addWidget(self.label_waitAfterReset_muS, 2, 0, 1, 1)
        self.doubleSpinBox_waitAfterReset_muS = QtWidgets.QDoubleSpinBox(self.advancedSeqSettings)
        self.doubleSpinBox_waitAfterReset_muS.setFrame(True)
        self.doubleSpinBox_waitAfterReset_muS.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.doubleSpinBox_waitAfterReset_muS.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.doubleSpinBox_waitAfterReset_muS.setKeyboardTracking(False)
        self.doubleSpinBox_waitAfterReset_muS.setDecimals(3)
        self.doubleSpinBox_waitAfterReset_muS.setMaximum(10000.0)
        self.doubleSpinBox_waitAfterReset_muS.setSingleStep(0.5)
        self.doubleSpinBox_waitAfterReset_muS.setObjectName("doubleSpinBox_waitAfterReset_muS")
        self.gridLayout_5.addWidget(self.doubleSpinBox_waitAfterReset_muS, 2, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.advancedSeqSettings)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setObjectName("label_2")
        self.gridLayout_5.addWidget(self.label_2, 0, 0, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.advancedSeqSettings)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_6.sizePolicy().hasHeightForWidth())
        self.label_6.setSizePolicy(sizePolicy)
        self.label_6.setObjectName("label_6")
        self.gridLayout_5.addWidget(self.label_6, 0, 1, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.advancedSeqSettings)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_13.sizePolicy().hasHeightForWidth())
        self.label_13.setSizePolicy(sizePolicy)
        self.label_13.setObjectName("label_13")
        self.gridLayout_5.addWidget(self.label_13, 0, 2, 1, 1)
        self.label_waitAfterReset_muS_set = QtWidgets.QLabel(self.advancedSeqSettings)
        self.label_waitAfterReset_muS_set.setObjectName("label_waitAfterReset_muS_set")
        self.gridLayout_5.addWidget(self.label_waitAfterReset_muS_set, 2, 2, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_5.addItem(spacerItem1, 3, 1, 1, 1)
        self.tabwidget.addTab(self.advancedSeqSettings, "")
        self.verticalLayout_4.addWidget(self.tabwidget)
        self.gridWidgetButtons = QtWidgets.QWidget(self.commonSequencerSettings)
        self.gridWidgetButtons.setMinimumSize(QtCore.QSize(366, 50))
        self.gridWidgetButtons.setObjectName("gridWidgetButtons")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.gridWidgetButtons)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.pushButtonResetToDefault = QtWidgets.QPushButton(self.gridWidgetButtons)
        self.pushButtonResetToDefault.setObjectName("pushButtonResetToDefault")
        self.gridLayout_3.addWidget(self.pushButtonResetToDefault, 0, 0, 1, 1)
        self.pushButton_confirm = QtWidgets.QPushButton(self.gridWidgetButtons)
        self.pushButton_confirm.setObjectName("pushButton_confirm")
        self.gridLayout_3.addWidget(self.pushButton_confirm, 0, 1, 1, 1)
        self.pushButton_cancel = QtWidgets.QPushButton(self.gridWidgetButtons)
        self.pushButton_cancel.setObjectName("pushButton_cancel")
        self.gridLayout_3.addWidget(self.pushButton_cancel, 0, 2, 1, 1)
        self.verticalLayout_4.addWidget(self.gridWidgetButtons)
        self.verticalLayout.addWidget(self.commonSequencerSettings)
        self.gridLayout_2.addLayout(self.verticalLayout, 0, 0, 1, 1)
        MainWindowTrackPars.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindowTrackPars)
        self.statusbar.setObjectName("statusbar")
        MainWindowTrackPars.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindowTrackPars)
        self.tabwidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindowTrackPars)
        MainWindowTrackPars.setTabOrder(self.doubleSpinBox_dacStartV, self.doubleSpinBox_dacStopV)
        MainWindowTrackPars.setTabOrder(self.doubleSpinBox_dacStopV, self.doubleSpinBox_dacStepSizeV)
        MainWindowTrackPars.setTabOrder(self.doubleSpinBox_dacStepSizeV, self.spinBox_nOfSteps)
        MainWindowTrackPars.setTabOrder(self.spinBox_nOfSteps, self.spinBox_nOfScans)
        MainWindowTrackPars.setTabOrder(self.spinBox_nOfScans, self.checkBox_invertScan)
        MainWindowTrackPars.setTabOrder(self.checkBox_invertScan, self.doubleSpinBox_postAccOffsetVolt)
        MainWindowTrackPars.setTabOrder(self.doubleSpinBox_postAccOffsetVolt, self.lineEdit_activePmtList)
        MainWindowTrackPars.setTabOrder(self.lineEdit_activePmtList, self.checkBox_colDirTrue)

    def retranslateUi(self, MainWindowTrackPars):
        _translate = QtCore.QCoreApplication.translate
        MainWindowTrackPars.setWindowTitle(_translate("MainWindowTrackPars", "track parameters"))
        self.label_nOfScans.setText(_translate("MainWindowTrackPars", "number of Scans"))
        self.label_invertScan.setText(_translate("MainWindowTrackPars", "invert scan direction"))
        self.label_dacStartV_set.setText(_translate("MainWindowTrackPars", "None"))
        self.label_activePmtList.setText(_translate("MainWindowTrackPars", "active PMT List"))
        self.label_postAccOffsetVolt.setText(_translate("MainWindowTrackPars", "post Acceleration \n"
"power supply voltage [V]"))
        self.label_postAccOffsetVolt_set.setText(_translate("MainWindowTrackPars", "None"))
        self.label_nOfScans_set.setText(_translate("MainWindowTrackPars", "None"))
        self.label_colDirTrue.setText(_translate("MainWindowTrackPars", "measured collinear?"))
        self.label_colDirTrue_set.setText(_translate("MainWindowTrackPars", "None"))
        self.label_5.setText(_translate("MainWindowTrackPars", "Set Value"))
        self.label_dacStartV.setText(_translate("MainWindowTrackPars", "DAC start  [V]"))
        self.label_4.setText(_translate("MainWindowTrackPars", "User Input"))
        self.label_postAccOffsetVoltControl.setText(_translate("MainWindowTrackPars", "post Acceleration \n"
"power supply"))
        self.label_dacStopV_set.setText(_translate("MainWindowTrackPars", "None"))
        self.label_dacStepSizeV.setText(_translate("MainWindowTrackPars", "DAC step size [V]"))
        self.label_dacStopV.setText(_translate("MainWindowTrackPars", "DAC stop [V]"))
        self.label_nOfSteps_set.setText(_translate("MainWindowTrackPars", "None"))
        self.label_dacStepSizeV_set.setText(_translate("MainWindowTrackPars", "None"))
        self.label_nOfSteps.setText(_translate("MainWindowTrackPars", "number of steps"))
        self.label_activePmtList_set.setText(_translate("MainWindowTrackPars", "None"))
        self.label_invertScan_set.setText(_translate("MainWindowTrackPars", "None"))
        self.label_3.setText(_translate("MainWindowTrackPars", "Parameter"))
        self.label_kepco_start.setText(_translate("MainWindowTrackPars", "None"))
        self.label_7.setText(_translate("MainWindowTrackPars", "DAC * 50"))
        self.label_kepco_stop.setText(_translate("MainWindowTrackPars", "None"))
        self.label_kepco_step.setText(_translate("MainWindowTrackPars", "None"))
        self.comboBox_postAccOffsetVoltControl.setItemText(0, _translate("MainWindowTrackPars", "Kepco"))
        self.comboBox_postAccOffsetVoltControl.setItemText(1, _translate("MainWindowTrackPars", "Heinzinger1"))
        self.comboBox_postAccOffsetVoltControl.setItemText(2, _translate("MainWindowTrackPars", "Heinzinger2"))
        self.comboBox_postAccOffsetVoltControl.setItemText(3, _translate("MainWindowTrackPars", "Heinzinger3"))
        self.pushButton_postAccOffsetVolt.setText(_translate("MainWindowTrackPars", "set volt"))
        self.label_postAccOffsetVoltControl_set.setText(_translate("MainWindowTrackPars", "None"))
        self.label_11.setText(_translate("MainWindowTrackPars", "readback\n"
"Voltage:"))
        self.tabwidget.setTabText(self.tabwidget.indexOf(self.seqComSet), _translate("MainWindowTrackPars", "common Sequencer Settings"))
        self.label_waitForKepco_muS_set.setText(_translate("MainWindowTrackPars", "None"))
        self.label_waitForKepco_muS.setText(_translate("MainWindowTrackPars", "pre scan wait [µs]"))
        self.label_waitAfterReset_muS.setText(_translate("MainWindowTrackPars", "wait after voltage reset [µs]"))
        self.label_2.setText(_translate("MainWindowTrackPars", "parameter"))
        self.label_6.setText(_translate("MainWindowTrackPars", "user input"))
        self.label_13.setText(_translate("MainWindowTrackPars", "set value"))
        self.label_waitAfterReset_muS_set.setText(_translate("MainWindowTrackPars", "None"))
        self.tabwidget.setTabText(self.tabwidget.indexOf(self.advancedSeqSettings), _translate("MainWindowTrackPars", "advanced Settings"))
        self.pushButtonResetToDefault.setText(_translate("MainWindowTrackPars", "reset to default"))
        self.pushButton_confirm.setText(_translate("MainWindowTrackPars", "confirm"))
        self.pushButton_cancel.setText(_translate("MainWindowTrackPars", "cancel"))

