"""
Created on 

@author: simkaufm

Module Description: Simple User Interface to display a set voltage and read it back
"""

from PyQt5 import QtWidgets
import time

from Interface.SetVoltageUi.Ui_SetVoltage import Ui_Dialog
import Application.Config as Cfg


class SetVoltageUi(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self, power_supply, new_voltage):
        QtWidgets.QDialog.__init__(self)
        self.power_supply = power_supply
        self.readback = None

        self.setupUi(self)

        self.setWindowTitle(self.power_supply)

        self.label_targetVoltage.setText(str(new_voltage))
        self.pushButton_ok.clicked.connect(self.ok)
        self.pushButton_refresh.clicked.connect(self.request_power_supply_status)

        self.request_power_supply_status()

        self.exec_()

    def request_power_supply_status(self):
        Cfg._main_instance.power_supply_status_request(self.power_supply)
        tries = 0
        while self.write_labels() and tries < 100:
            tries += 1
            time.sleep(0.005)

    def write_labels(self):
        power_status_dict = Cfg._main_instance.requested_power_supply_status
        if Cfg._main_instance.requested_power_supply_status is not None:
            self.label_lastSetVolt.setText(str(power_status_dict.get('programmedVoltage')))
            self.label_lastVoltageSetAt.setText(power_status_dict.get('voltageSetTime'))
            self.readback = str(power_status_dict.get('readBackVolt'))
            self.label_voltReadBack.setText(self.readback)
            Cfg._main_instance.requested_power_supply_status = None
            return False
        return True

    def ok(self):
        self.close()
