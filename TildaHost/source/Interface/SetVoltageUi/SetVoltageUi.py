"""
Created on 

@author: simkaufm

Module Description: Simple User Interface to display a set voltage and read it back
"""

from PyQt5 import QtWidgets

from Interface.SetVoltageUi.Ui_SetVoltage import Ui_Dialog
import Application.Config as Cfg


class SetVoltageUi(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self, power_supply, new_voltage):
        QtWidgets.QDialog.__init__(self)
        self.power_supply = power_supply
        self.main = Cfg._main_instance
        self.readback = None

        self.setupUi(self)

        self.setWindowTitle(self.power_supply)

        self.label_targetVoltage.setText(str(new_voltage))
        self.pushButton_ok.clicked.connect(self.ok)
        self.pushButton_refresh.clicked.connect(self.request_power_supply_status)

        self.request_power_supply_status()
        self.exec_()

    def request_power_supply_status(self):
        power_status_dict = self.main.power_supply_status_request(self.power_supply)
        if power_status_dict is None:
            power_status_dict = {}
        self.label_lastSetVolt.setText(str(power_status_dict.get('programmedVoltage')))
        self.label_lastVoltageSetAt.setText(power_status_dict.get('voltageSetTime'))
        self.readback = str(power_status_dict.get('readBackVolt'))
        self.label_voltReadBack.setText(self.readback)

    def ok(self):
        self.close()
