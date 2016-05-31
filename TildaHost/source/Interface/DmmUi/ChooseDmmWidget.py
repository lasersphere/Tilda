"""
Created on 

@author: simkaufm

Module Description:
"""

from PyQt5 import QtWidgets, QtGui
import sys

from Interface.DmmUi.Ui_ChooseDmmWidget import Ui_Form


class ChooseDmmWidget(QtWidgets.QWidget, Ui_Form):
    def __init__(self, callback, dmm_names):
        super(ChooseDmmWidget, self).__init__()
        self.setupUi(self)
        self.callback = callback
        self.comboBox_choose_dmm.addItems(dmm_names)

        self.pushButton_initialize.clicked.connect(self.init_dmm)

    def init_dmm(self):
        dev_name = self.comboBox_choose_dmm.currentText()
        dev_address = self.lineEdit_address_dmm.text()
        self.callback.emit((dev_name, dev_address))
