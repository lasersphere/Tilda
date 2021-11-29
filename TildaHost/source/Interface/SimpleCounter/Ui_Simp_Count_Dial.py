# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_Simp_Count_Dial.ui'
#
# Created by: PyQt5 UI code generator 5.5.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog_simpleCounterControl(object):
    def setupUi(self, Dialog_simpleCounterControl):
        Dialog_simpleCounterControl.setObjectName("Dialog_simpleCounterControl")
        Dialog_simpleCounterControl.resize(555, 480)
        Dialog_simpleCounterControl.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.label_5 = QtWidgets.QLabel(Dialog_simpleCounterControl)
        self.label_5.setEnabled(True)
        self.label_5.setGeometry(QtCore.QRect(9, 9, 541, 41))
        self.label_5.setObjectName("label_5")
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog_simpleCounterControl)
        self.buttonBox.setGeometry(QtCore.QRect(380, 430, 156, 23))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.widget = QtWidgets.QWidget(Dialog_simpleCounterControl)
        self.widget.setGeometry(QtCore.QRect(10, 60, 531, 191))
        self.widget.setObjectName("widget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_3 = QtWidgets.QLabel(self.widget)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 0, 1, 1)
        self.spinBox_plotpoints = QtWidgets.QSpinBox(self.widget)
        self.spinBox_plotpoints.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.spinBox_plotpoints.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.spinBox_plotpoints.setKeyboardTracking(False)
        self.spinBox_plotpoints.setMaximum(10000)
        self.spinBox_plotpoints.setObjectName("spinBox_plotpoints")
        self.gridLayout.addWidget(self.spinBox_plotpoints, 2, 1, 1, 1)
        self.lineEdit_act_pmts = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_act_pmts.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.lineEdit_act_pmts.setObjectName("lineEdit_act_pmts")
        self.gridLayout.addWidget(self.lineEdit_act_pmts, 0, 1, 1, 1)
        self.label_act_pmts_set = QtWidgets.QLabel(self.widget)
        self.label_act_pmts_set.setObjectName("label_act_pmts_set")
        self.gridLayout.addWidget(self.label_act_pmts_set, 0, 2, 1, 1)
        self.label_plotpoints_set = QtWidgets.QLabel(self.widget)
        self.label_plotpoints_set.setObjectName("label_plotpoints_set")
        self.gridLayout.addWidget(self.label_plotpoints_set, 2, 2, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.horizontalLayout_trigger = QtWidgets.QHBoxLayout()
        self.horizontalLayout_trigger.setObjectName("horizontalLayout_trigger")
        self.verticalLayout_triggerSelect = QtWidgets.QVBoxLayout()
        self.verticalLayout_triggerSelect.setObjectName("verticalLayout_triggerSelect")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setObjectName("label")
        self.verticalLayout_triggerSelect.addWidget(self.label)
        self.comboBox_triggerSelect = QtWidgets.QComboBox(self.widget)
        self.comboBox_triggerSelect.setObjectName("comboBox_triggerSelect")
        self.verticalLayout_triggerSelect.addWidget(self.comboBox_triggerSelect)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_triggerSelect.addItem(spacerItem)
        self.horizontalLayout_trigger.addLayout(self.verticalLayout_triggerSelect)
        self.verticalLayout_trigger = QtWidgets.QVBoxLayout()
        self.verticalLayout_trigger.setObjectName("verticalLayout_trigger")
        self.widget_trigger_place_holder = QtWidgets.QWidget(self.widget)
        self.widget_trigger_place_holder.setObjectName("widget_trigger_place_holder")
        self.verticalLayout_trigger.addWidget(self.widget_trigger_place_holder)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_trigger.addItem(spacerItem1)
        self.horizontalLayout_trigger.addLayout(self.verticalLayout_trigger)
        self.verticalLayout.addLayout(self.horizontalLayout_trigger)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem2)
        self.widget_timeGate_place_holder = QtWidgets.QWidget(self.widget)
        self.widget_timeGate_place_holder.setObjectName("widget_timeGate_place_holder")
        self.verticalLayout.addWidget(self.widget_timeGate_place_holder)

        self.retranslateUi(Dialog_simpleCounterControl)
        self.buttonBox.accepted.connect(Dialog_simpleCounterControl.accept)
        self.buttonBox.rejected.connect(Dialog_simpleCounterControl.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog_simpleCounterControl)

    def retranslateUi(self, Dialog_simpleCounterControl):
        _translate = QtCore.QCoreApplication.translate
        Dialog_simpleCounterControl.setWindowTitle(_translate("Dialog_simpleCounterControl", "SimpleCounterControl"))
        self.label_5.setText(_translate("Dialog_simpleCounterControl", "<html><head/><body><p>select active pmt\'s &amp; number of datapoint<br/>choose trigger</p></body></html>"))
        self.label_3.setText(_translate("Dialog_simpleCounterControl", "datapoints"))
        self.label_2.setText(_translate("Dialog_simpleCounterControl", "active pmts"))
        self.label_act_pmts_set.setText(_translate("Dialog_simpleCounterControl", "TextLabel"))
        self.label_plotpoints_set.setText(_translate("Dialog_simpleCounterControl", "TextLabel"))
        self.label.setText(_translate("Dialog_simpleCounterControl", "trigger"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog_simpleCounterControl = QtWidgets.QDialog()
    ui = Ui_Dialog_simpleCounterControl()
    ui.setupUi(Dialog_simpleCounterControl)
    Dialog_simpleCounterControl.show()
    sys.exit(app.exec_())

