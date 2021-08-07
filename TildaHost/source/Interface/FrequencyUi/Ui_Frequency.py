# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_Frequency.ui'
#
# Created by: PyQt5 UI code generator 5.5
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Frequency(object):
    def setupUi(self, Frequency):
        Frequency.setObjectName("Frequency")
        Frequency.resize(400, 300)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("../icons/Tilda256.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Frequency.setWindowIcon(icon)
        self.buttonBox = QtWidgets.QDialogButtonBox(Frequency)
        self.buttonBox.setGeometry(QtCore.QRect(30, 240, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.le_arit = QtWidgets.QLineEdit(Frequency)
        self.le_arit.setGeometry(QtCore.QRect(10, 30, 371, 20))
        self.le_arit.setObjectName("le_arit")
        self.label = QtWidgets.QLabel(Frequency)
        self.label.setGeometry(QtCore.QRect(10, 10, 121, 16))
        self.label.setObjectName("label")
        self.horizontalLayoutWidget = QtWidgets.QWidget(Frequency)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 80, 371, 151))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.list_frequencies = QtWidgets.QListWidget(self.horizontalLayoutWidget)
        self.list_frequencies.setObjectName("list_frequencies")
        self.horizontalLayout.addWidget(self.list_frequencies)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.pb_add = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pb_add.setObjectName("pb_add")
        self.verticalLayout.addWidget(self.pb_add)
        self.pb_remove = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pb_remove.setObjectName("pb_remove")
        self.verticalLayout.addWidget(self.pb_remove)
        self.pb_edit = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pb_edit.setObjectName("pb_edit")
        self.verticalLayout.addWidget(self.pb_edit)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.label_2 = QtWidgets.QLabel(Frequency)
        self.label_2.setGeometry(QtCore.QRect(10, 50, 371, 21))
        self.label_2.setObjectName("label_2")

        self.retranslateUi(Frequency)
        self.buttonBox.accepted.connect(Frequency.accept)
        self.buttonBox.rejected.connect(Frequency.reject)
        QtCore.QMetaObject.connectSlotsByName(Frequency)

    def retranslateUi(self, Frequency):
        _translate = QtCore.QCoreApplication.translate
        Frequency.setWindowTitle(_translate("Frequency", "Frequency"))
        self.label.setText(_translate("Frequency", "Frequency Arithmetic"))
        self.pb_add.setText(_translate("Frequency", "Add Frequency"))
        self.pb_remove.setText(_translate("Frequency", "Remove"))
        self.pb_edit.setText(_translate("Frequency", "Edit"))
        self.label_2.setText(_translate("Frequency", "Given Frequency names must correspond to the name used in the arithmetic."))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Frequency = QtWidgets.QDialog()
    ui = Ui_Frequency()
    ui.setupUi(Frequency)
    Frequency.show()
    sys.exit(app.exec_())

