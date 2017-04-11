"""
Created on 

@author: simkaufm

Module Description:
"""

from PyQt5 import QtWidgets

from Interface.DmmUi.Ui_ChooseDmmWidget import Ui_Form


class ChooseDmmWidget(QtWidgets.QWidget, Ui_Form):
    def __init__(self, callback, dmm_names):
        super(ChooseDmmWidget, self).__init__()
        self.setupUi(self)
        self.callback = callback
        self.comboBox_choose_dmm.addItems(dmm_names)
        self.check_standard_address(self.comboBox_choose_dmm.currentText())

        self.pushButton_initialize.clicked.connect(self.init_dmm)
        self.comboBox_choose_dmm.currentTextChanged.connect(self.check_standard_address)

    def init_dmm(self):
        dev_name = self.comboBox_choose_dmm.currentText()
        dev_address = self.lineEdit_address_dmm.text()
        self.callback.emit((dev_name, dev_address))

    def check_standard_address(self, type_str):
        """
        resolve the standard address and write it to the line edit

        for the future this should be stored somweher else than in the gui.
        """
        addr = 'please insert address'
        if type_str == 'Ni4071':
            addr = 'PXI1Slot5'
        elif type_str == 'dummy':
            addr = 'somewhere'
        elif type_str == 'Agilent_34461A':
            addr = 'COLLAPSAGILENT01'
        elif type_str == 'Agilent_34401A':
            addr = 'com1'
        elif type_str == 'Agilent_M918x':
            addr = 'PXI6::15::INSTR'
        self.lineEdit_address_dmm.setText(addr)
        self.lineEdit_address_dmm.setToolTip(' for COLLAPS:\n'
                                             'COLLAPSAGILENT01 - 137.138.135.84 (not fixed ?)'
                                             ' - Isolde hall COLLAPS DAQ area\n'
                                             'COLLAPSAGILENT02 - 137.138.135.94 (not fixed ?)'
                                             ' - ISCOOL rack\n'
                                             )
