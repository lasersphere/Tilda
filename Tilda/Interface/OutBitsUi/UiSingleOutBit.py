"""
Created on 

@author: simkaufm

Module Description:  Module for editing a single outbit command which is already existing in cmd_dict
"""

import logging
from PyQt5 import QtWidgets


from Tilda.Interface.OutBitsUi.Ui_SingleOutBit import Ui_EditOutbit


class UiSingleOutBit(QtWidgets.QDialog, Ui_EditOutbit):
    def __init__(self, parent, cmd_dict, outb_name, lis_index, cmd):
        super(UiSingleOutBit, self).__init__(parent)
        self.setupUi(self)

        self.cmd_dict = cmd_dict
        self.list_index = None

        """ fill comboboxes """
        self.setup_comboboxes()

        """ set with existing vals """
        bits = ['outbit0', 'outbit1', 'outbit2']

        if outb_name in bits:
            self.list_index = lis_index
            self.comboBox_bit_sel.setCurrentText(outb_name)
            self.comboBox_toggle_on_off.setCurrentText(cmd[0])
            self.comboBox_scan_step.setCurrentText(cmd[1])
            self.spinBox_scan_step_num.setValue(cmd[2])
        self.show()
        logging.info('opened UiSingleOutBit gui for %s with cmd %s' % (outb_name, str(cmd)))

    def setup_comboboxes(self):
        bits = ['outbit0', 'outbit1', 'outbit2']
        toggle_on_offs = ['toggle', 'on', 'off']
        scan_steps = ['scan', 'step']

        self.comboBox_bit_sel.clear()
        self.comboBox_toggle_on_off.clear()
        self.comboBox_scan_step.clear()

        self.comboBox_bit_sel.addItems(bits)
        self.comboBox_toggle_on_off.addItems(toggle_on_offs)
        self.comboBox_scan_step.addItems(scan_steps)

    def get_cmd(self):
        """ read the gui for the current cmd and return this"""
        outb_name = self.comboBox_bit_sel.currentText()
        cmd = (
            self.comboBox_toggle_on_off.currentText(),
            self.comboBox_scan_step.currentText(),
            self.spinBox_scan_step_num.value()
        )
        return outb_name, cmd


