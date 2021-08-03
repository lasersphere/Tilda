# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_Add_Freq.ui'
#
# Created by: PyQt5 UI code generator 5.5.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Add_Freq(object):
    def setupUi(self, Add_Freq):
        Add_Freq.setObjectName("Add_Freq")
        Add_Freq.resize(400, 115)
        self.buttonBox = QtWidgets.QDialogButtonBox(Add_Freq)
        self.buttonBox.setGeometry(QtCore.QRect(30, 70, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayoutWidget = QtWidgets.QWidget(Add_Freq)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(10, 10, 371, 51))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setObjectName("gridLayout")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.gridLayout.addWidget(self.lineEdit_2, 1, 2, 1, 1)
        self.lineEdit = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout.addWidget(self.lineEdit, 1, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 2, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 1, 3, 1, 1)

        self.retranslateUi(Add_Freq)
        self.buttonBox.accepted.connect(Add_Freq.accept)
        self.buttonBox.rejected.connect(Add_Freq.reject)
        QtCore.QMetaObject.connectSlotsByName(Add_Freq)

    def retranslateUi(self, Add_Freq):
        _translate = QtCore.QCoreApplication.translate
        Add_Freq.setWindowTitle(_translate("Add_Freq", "Dialog"))
        self.label.setText(_translate("Add_Freq", "Name"))
        self.label_2.setText(_translate("Add_Freq", "Value"))
        self.label_3.setText(_translate("Add_Freq", "MHz"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Add_Freq = QtWidgets.QDialog()
    ui = Ui_Add_Freq()
    ui.setupUi(Add_Freq)
    Add_Freq.show()
    sys.exit(app.exec_())

