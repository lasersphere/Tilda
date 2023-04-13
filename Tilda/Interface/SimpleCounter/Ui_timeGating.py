# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_timeGating.ui'
#
# Created by: PyQt5 UI code generator 5.5.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_timeGating(object):
    def setupUi(self, timeGating):
        timeGating.setObjectName("timeGating")
        timeGating.resize(400, 126)
        self.gridLayout = QtWidgets.QGridLayout(timeGating)
        self.gridLayout.setObjectName("gridLayout")
        self.label_4 = QtWidgets.QLabel(timeGating)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 1, 2, 1, 1)
        self.label = QtWidgets.QLabel(timeGating)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(timeGating)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.lineEdit_mid_tof = QtWidgets.QLineEdit(timeGating)
        self.lineEdit_mid_tof.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.lineEdit_mid_tof.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.lineEdit_mid_tof.setObjectName("lineEdit_mid_tof")
        self.gridLayout.addWidget(self.lineEdit_mid_tof, 2, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(timeGating)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 1, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(timeGating)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 2, 0, 1, 1)
        self.label_mid_tof = QtWidgets.QLabel(timeGating)
        self.label_mid_tof.setObjectName("label_mid_tof")
        self.gridLayout.addWidget(self.label_mid_tof, 2, 2, 1, 1)
        self.label_7 = QtWidgets.QLabel(timeGating)
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 3, 0, 1, 1)
        self.lineEdit_gate_width = QtWidgets.QLineEdit(timeGating)
        self.lineEdit_gate_width.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.lineEdit_gate_width.setObjectName("lineEdit_gate_width")
        self.gridLayout.addWidget(self.lineEdit_gate_width, 3, 1, 1, 1)
        self.label_gate_width = QtWidgets.QLabel(timeGating)
        self.label_gate_width.setObjectName("label_gate_width")
        self.gridLayout.addWidget(self.label_gate_width, 3, 2, 1, 1)

        self.retranslateUi(timeGating)
        QtCore.QMetaObject.connectSlotsByName(timeGating)

    def retranslateUi(self, timeGating):
        _translate = QtCore.QCoreApplication.translate
        timeGating.setWindowTitle(_translate("timeGating", "timeGating"))
        self.label_4.setText(_translate("timeGating", "Set Value"))
        self.label.setText(_translate("timeGating", "time gate settings"))
        self.label_2.setText(_translate("timeGating", "Parameter"))
        self.lineEdit_mid_tof.setText(_translate("timeGating", "0.00"))
        self.label_3.setText(_translate("timeGating", "User Input"))
        self.label_5.setText(_translate("timeGating", "softw. mid TOF / us"))
        self.label_mid_tof.setText(_translate("timeGating", "None"))
        self.label_7.setText(_translate("timeGating", "softw. gate width / us"))
        self.lineEdit_gate_width.setText(_translate("timeGating", "0.00"))
        self.label_gate_width.setText(_translate("timeGating", "None"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    timeGating = QtWidgets.QWidget()
    ui = Ui_timeGating()
    ui.setupUi(timeGating)
    timeGating.show()
    sys.exit(app.exec_())

