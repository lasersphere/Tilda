# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_Options.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog_Options(object):
    def setupUi(self, Dialog_Options):
        Dialog_Options.setObjectName("Dialog_Options")
        Dialog_Options.resize(555, 568)
        Dialog_Options.setModal(False)
        self.verticalLayout = QtWidgets.QVBoxLayout(Dialog_Options)
        self.verticalLayout.setObjectName("verticalLayout")
        self.tabWidget = QtWidgets.QTabWidget(Dialog_Options)
        self.tabWidget.setObjectName("tabWidget")
        self.tab_generalSettings = QtWidgets.QWidget()
        self.tab_generalSettings.setObjectName("tab_generalSettings")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.tab_generalSettings)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.groupBox_preScan = QtWidgets.QGroupBox(self.tab_generalSettings)
        self.groupBox_preScan.setObjectName("groupBox_preScan")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.groupBox_preScan)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label = QtWidgets.QLabel(self.groupBox_preScan)
        self.label.setScaledContents(False)
        self.label.setWordWrap(True)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        self.doubleSpinBox_preScanTimeout = QtWidgets.QDoubleSpinBox(self.groupBox_preScan)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.doubleSpinBox_preScanTimeout.sizePolicy().hasHeightForWidth())
        self.doubleSpinBox_preScanTimeout.setSizePolicy(sizePolicy)
        self.doubleSpinBox_preScanTimeout.setDecimals(1)
        self.doubleSpinBox_preScanTimeout.setMinimum(0.0)
        self.doubleSpinBox_preScanTimeout.setMaximum(1000.0)
        self.doubleSpinBox_preScanTimeout.setProperty("value", 60.0)
        self.doubleSpinBox_preScanTimeout.setObjectName("doubleSpinBox_preScanTimeout")
        self.horizontalLayout_2.addWidget(self.doubleSpinBox_preScanTimeout)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.verticalLayout_2.addWidget(self.groupBox_preScan)
        self.groupBox_Connect = QtWidgets.QGroupBox(self.tab_generalSettings)
        self.groupBox_Connect.setObjectName("groupBox_Connect")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.groupBox_Connect)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.link_openFpgaConfig = QtWidgets.QCommandLinkButton(self.groupBox_Connect)
        self.link_openFpgaConfig.setObjectName("link_openFpgaConfig")
        self.verticalLayout_4.addWidget(self.link_openFpgaConfig)
        self.line = QtWidgets.QFrame(self.groupBox_Connect)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout_4.addWidget(self.line)
        self.link_openTritonConfig = QtWidgets.QCommandLinkButton(self.groupBox_Connect)
        self.link_openTritonConfig.setObjectName("link_openTritonConfig")
        self.verticalLayout_4.addWidget(self.link_openTritonConfig)
        self.checkBox_disableTritonLink = QtWidgets.QCheckBox(self.groupBox_Connect)
        self.checkBox_disableTritonLink.setObjectName("checkBox_disableTritonLink")
        self.verticalLayout_4.addWidget(self.checkBox_disableTritonLink)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setContentsMargins(-1, 0, -1, -1)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_3 = QtWidgets.QLabel(self.groupBox_Connect)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_3.addWidget(self.label_3)
        self.spinBox_trritinReadInterval = QtWidgets.QSpinBox(self.groupBox_Connect)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.spinBox_trritinReadInterval.sizePolicy().hasHeightForWidth())
        self.spinBox_trritinReadInterval.setSizePolicy(sizePolicy)
        self.spinBox_trritinReadInterval.setMaximum(9999)
        self.spinBox_trritinReadInterval.setProperty("value", 100)
        self.spinBox_trritinReadInterval.setObjectName("spinBox_trritinReadInterval")
        self.horizontalLayout_3.addWidget(self.spinBox_trritinReadInterval)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem1)
        self.verticalLayout_4.addLayout(self.horizontalLayout_3)
        self.verticalLayout_2.addWidget(self.groupBox_Connect)
        self.groupBox_scanFinished = QtWidgets.QGroupBox(self.tab_generalSettings)
        self.groupBox_scanFinished.setEnabled(True)
        self.groupBox_scanFinished.setAutoFillBackground(False)
        self.groupBox_scanFinished.setFlat(False)
        self.groupBox_scanFinished.setCheckable(True)
        self.groupBox_scanFinished.setObjectName("groupBox_scanFinished")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.groupBox_scanFinished)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_scanFinishedWin = QtWidgets.QLabel(self.groupBox_scanFinished)
        self.label_scanFinishedWin.setObjectName("label_scanFinishedWin")
        self.verticalLayout_3.addWidget(self.label_scanFinishedWin)
        self.checkBox_playSound = QtWidgets.QCheckBox(self.groupBox_scanFinished)
        self.checkBox_playSound.setObjectName("checkBox_playSound")
        self.verticalLayout_3.addWidget(self.checkBox_playSound)
        self.link_openSoundsFolder = QtWidgets.QCommandLinkButton(self.groupBox_scanFinished)
        self.link_openSoundsFolder.setObjectName("link_openSoundsFolder")
        self.verticalLayout_3.addWidget(self.link_openSoundsFolder)
        self.pushButton_chooseSoundsFolder = QtWidgets.QPushButton(self.groupBox_scanFinished)
        self.pushButton_chooseSoundsFolder.setObjectName("pushButton_chooseSoundsFolder")
        self.verticalLayout_3.addWidget(self.pushButton_chooseSoundsFolder)
        self.verticalLayout_2.addWidget(self.groupBox_scanFinished)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem2)
        self.tabWidget.addTab(self.tab_generalSettings, "")
        self.tab_polliFit = QtWidgets.QWidget()
        self.tab_polliFit.setObjectName("tab_polliFit")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.tab_polliFit)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.checkBox_guessOffset = QtWidgets.QCheckBox(self.tab_polliFit)
        self.checkBox_guessOffset.setObjectName("checkBox_guessOffset")
        self.verticalLayout_6.addWidget(self.checkBox_guessOffset)
        self.tabWidget.addTab(self.tab_polliFit, "")
        self.tab_special = QtWidgets.QWidget()
        self.tab_special.setObjectName("tab_special")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.tab_special)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.checkBox_enableRoc = QtWidgets.QCheckBox(self.tab_special)
        self.checkBox_enableRoc.setObjectName("checkBox_enableRoc")
        self.verticalLayout_5.addWidget(self.checkBox_enableRoc)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.spinBox_xmlResolution = QtWidgets.QSpinBox(self.tab_special)
        self.spinBox_xmlResolution.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.spinBox_xmlResolution.sizePolicy().hasHeightForWidth())
        self.spinBox_xmlResolution.setSizePolicy(sizePolicy)
        self.spinBox_xmlResolution.setPrefix("")
        self.spinBox_xmlResolution.setMinimum(10)
        self.spinBox_xmlResolution.setMaximum(100000)
        self.spinBox_xmlResolution.setObjectName("spinBox_xmlResolution")
        self.horizontalLayout.addWidget(self.spinBox_xmlResolution)
        self.label_2 = QtWidgets.QLabel(self.tab_special)
        self.label_2.setEnabled(False)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.checkBox_autoResolution = QtWidgets.QCheckBox(self.tab_special)
        self.checkBox_autoResolution.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.checkBox_autoResolution.sizePolicy().hasHeightForWidth())
        self.checkBox_autoResolution.setSizePolicy(sizePolicy)
        self.checkBox_autoResolution.setObjectName("checkBox_autoResolution")
        self.horizontalLayout.addWidget(self.checkBox_autoResolution)
        self.verticalLayout_5.addLayout(self.horizontalLayout)
        self.tabWidget.addTab(self.tab_special, "")
        self.verticalLayout.addWidget(self.tabWidget)
        self.buttonBox_okCancel = QtWidgets.QDialogButtonBox(Dialog_Options)
        self.buttonBox_okCancel.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox_okCancel.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.RestoreDefaults|QtWidgets.QDialogButtonBox.SaveAll)
        self.buttonBox_okCancel.setObjectName("buttonBox_okCancel")
        self.verticalLayout.addWidget(self.buttonBox_okCancel)

        self.retranslateUi(Dialog_Options)
        self.tabWidget.setCurrentIndex(0)
        self.buttonBox_okCancel.accepted.connect(Dialog_Options.accept)
        self.buttonBox_okCancel.rejected.connect(Dialog_Options.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog_Options)

    def retranslateUi(self, Dialog_Options):
        _translate = QtCore.QCoreApplication.translate
        Dialog_Options.setWindowTitle(_translate("Dialog_Options", "TILDA Options"))
        self.groupBox_preScan.setTitle(_translate("Dialog_Options", "PRE SCAN"))
        self.label.setToolTip(_translate("Dialog_Options", "The pre scan timeout is the maximum time for any pre scan measurement. If not all measurements are completed within this time, the measurement is started anyhow."))
        self.label.setText(_translate("Dialog_Options", "Pre scan timeout:"))
        self.doubleSpinBox_preScanTimeout.setToolTip(_translate("Dialog_Options", "The pre scan timeout is the maximum time for any pre scan measurement. If not all measurements are completed within this time, the measurement is started anyhow."))
        self.doubleSpinBox_preScanTimeout.setSuffix(_translate("Dialog_Options", " s"))
        self.groupBox_Connect.setTitle(_translate("Dialog_Options", "CONNECT"))
        self.link_openFpgaConfig.setText(_translate("Dialog_Options", "...\\Driver\\DataAcquisitionFpga\\fpga_config.xml"))
        self.link_openFpgaConfig.setDescription(_translate("Dialog_Options", "Open fpga config to change the FPGA configuration for this system"))
        self.link_openTritonConfig.setText(_translate("Dialog_Options", "...\\Driver\\TritonListener\\TritonConfig.py"))
        self.link_openTritonConfig.setDescription(_translate("Dialog_Options", "Open config file to change the Triton configuration"))
        self.checkBox_disableTritonLink.setText(_translate("Dialog_Options", "DISABLE all Triton functionality (\"local\")"))
        self.label_3.setText(_translate("Dialog_Options", "Triton reading interval:"))
        self.spinBox_trritinReadInterval.setSuffix(_translate("Dialog_Options", " ms"))
        self.groupBox_scanFinished.setTitle(_translate("Dialog_Options", "SCAN FINISHED WINDOW"))
        self.label_scanFinishedWin.setText(_translate("Dialog_Options", "Show the green scan finished window aftger a successfull scan."))
        self.checkBox_playSound.setText(_translate("Dialog_Options", "Play random sound after each successful scan"))
        self.link_openSoundsFolder.setText(_translate("Dialog_Options", "...\\Interface\\Sounds"))
        self.link_openSoundsFolder.setDescription(_translate("Dialog_Options", "Go to Folder to add or remove sounds from the random rotation."))
        self.pushButton_chooseSoundsFolder.setText(_translate("Dialog_Options", "Choose Sounds Folder"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_generalSettings), _translate("Dialog_Options", "General"))
        self.checkBox_guessOffset.setText(_translate("Dialog_Options", "Guess Offset"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_polliFit), _translate("Dialog_Options", "PolliFit"))
        self.checkBox_enableRoc.setText(_translate("Dialog_Options", "Enable ROC Analysis Pipeline"))
        self.spinBox_xmlResolution.setSuffix(_translate("Dialog_Options", " ns"))
        self.label_2.setText(_translate("Dialog_Options", "Maximum xml-resolution (raw data always 10ns)"))
        self.checkBox_autoResolution.setToolTip(_translate("Dialog_Options", "Let TILDA estimate the array size and limit the resolution accordingly to avoid MEMORY erros. "))
        self.checkBox_autoResolution.setText(_translate("Dialog_Options", "Adapt Automatically"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_special), _translate("Dialog_Options", "Special"))
