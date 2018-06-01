"""
Created on 

@author: simkaufm

Module Description:
"""

from PyQt5 import QtWidgets
import logging

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
        self.lineEdit_address_dmm.textChanged.connect(self.check_valid_addr)

    def init_dmm(self):
        self.check_valid_addr()
        dev_name = self.comboBox_choose_dmm.currentText()
        dev_address = self.lineEdit_address_dmm.text()
        logging.debug('emitting %s, from %s, value is %s'
                      % ('callback',
                         'Interface.DmmUi.ChooseDmmWidget.ChooseDmmWidget#init_dmm',
                         str((dev_name, dev_address))))
        self.callback.emit((dev_name, dev_address))

    def check_standard_address(self, type_str):
        """
        resolve the standard address and write it to the line edit
        """
        default_addr_dict = {
            'Ni4071': 'PXI1Slot5',
            'dummy': 'somewhere',
            'Agilent_34461A': 'COLLAPSAGILENT01',
            'Agilent_34401A': 'com1',
            'Agilent_3458A': 'GPIB0..22..INSTR',
            'Agilent_M918x': 'PXI6..15..INSTR'
        }
        addr = default_addr_dict.get(type_str, 'please insert address')
        self.lineEdit_address_dmm.setText(addr)
        self.lineEdit_address_dmm.setToolTip(' for COLLAPS:\n'
                                             'COLLAPSAGILENT01 - 137.138.135.84 (not fixed ?)'
                                             ' - Isolde hall COLLAPS DAQ area\n'
                                             'COLLAPSAGILENT02 - 137.138.135.94 (not fixed ?)'
                                             ' - ISCOOL rack\n'
                                             )

    def check_valid_addr(self):
        to_check = self.lineEdit_address_dmm.text()
        if ':' in to_check:
            # colons in the address would go into the name of the dmm and
            # cause problems with the xml namespace when creating a file.
            warn = QtWidgets.QMessageBox.warning(
                self, 'warning',
                'colons not allowed in address!'
                'Use dots or something and convert in driver file of multimeter.\n\n'
                'Will correct to dot now.')
            # print('colons not allowed in address! Use dots or something and convert in driver file of multimeter')
            corrected = to_check.replace(':', '.')
            self.lineEdit_address_dmm.setText(corrected)
