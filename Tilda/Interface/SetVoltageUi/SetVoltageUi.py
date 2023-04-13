"""
Created on 

@author: simkaufm

Module Description: Simple User Interface to display a set voltage and read it back
"""

from PyQt5 import QtWidgets

from Tilda.Interface.SetVoltageUi.Ui_SetVoltage import Ui_Dialog
import Tilda.Application.Config as Cfg


class SetVoltageUi(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self, power_supply, new_voltage, call_back_sig):
        QtWidgets.QDialog.__init__(self)
        self.power_supply = power_supply
        self.call_back_sig = call_back_sig
        self.call_back_sig.connect(self.write_labels)

        self.setupUi(self)

        self.setWindowTitle(self.power_supply)

        self.label_targetVoltage.setText(str(new_voltage))
        self.pushButton_ok.clicked.connect(self.ok)
        self.pushButton_refresh.clicked.connect(self.request_power_supply_status)

        self.exec_()

    def request_power_supply_status(self):
        Cfg._main_instance.power_supply_status(self.power_supply, self.call_back_sig)

    def write_labels(self, power_status_dict):
        single_pwr_sup_dict = power_status_dict.get(self.power_supply)
        if single_pwr_sup_dict is not None:
            self.label_lastSetVolt.setText(str(single_pwr_sup_dict.get('programmedVoltage', 0)))
            self.label_lastVoltageSetAt.setText(single_pwr_sup_dict.get('voltageSetTime', 'None'))
            self.label_voltReadBack.setText(str(single_pwr_sup_dict.get('readBackVolt', 'None')))

    def ok(self):
        # self.call_back_sig.disconnect()
        self.close()
