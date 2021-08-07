# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_Add_Freq.ui'
#
# Created by: PyQt5 UI code generator 5.5
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Add_Freq(object):
    def setupUi(self, Add_Freq):
        Add_Freq.setObjectName("Add_Freq")
        Add_Freq.resize(400, 148)
        self.gridLayoutWidget = QtWidgets.QWidget(Add_Freq)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(10, 10, 371, 61))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setObjectName("gridLayout")
        self.le_value = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.le_value.setObjectName("le_value")
        self.gridLayout.addWidget(self.le_value, 1, 2, 1, 1)
        self.le_name = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.le_name.setObjectName("le_name")
        self.gridLayout.addWidget(self.le_name, 1, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 2, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 1, 3, 1, 1)
        self.pb_add = QtWidgets.QPushButton(Add_Freq)
        self.pb_add.setGeometry(QtCore.QRect(190, 100, 93, 28))
        self.pb_add.setObjectName("pb_add")
        self.pb_cancel = QtWidgets.QPushButton(Add_Freq)
        self.pb_cancel.setGeometry(QtCore.QRect(290, 100, 93, 28))
        self.pb_cancel.setObjectName("pb_cancel")

        self.retranslateUi(Add_Freq)
        QtCore.QMetaObject.connectSlotsByName(Add_Freq)

    def retranslateUi(self, Add_Freq):
        _translate = QtCore.QCoreApplication.translate
        Add_Freq.setWindowTitle(_translate("Add_Freq", "Add Frequency"))
        self.label.setText(_translate("Add_Freq", "Name"))
        self.label_2.setText(_translate("Add_Freq", "Value"))
        self.label_3.setText(_translate("Add_Freq", "MHz"))
        self.pb_add.setText(_translate("Add_Freq", "OK"))
        self.pb_cancel.setText(_translate("Add_Freq", "Cancel"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Add_Freq = QtWidgets.QDialog()
    ui = Ui_Add_Freq()
    ui.setupUi(Add_Freq)
    Add_Freq.show()
    sys.exit(app.exec_())

