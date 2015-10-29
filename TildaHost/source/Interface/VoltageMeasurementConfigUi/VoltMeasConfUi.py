"""

Created on '29.10.2015'

@author:'simkaufm'

"""

from Interface.VoltageMeasurementConfigUi.Ui_VoltMeasConf import Ui_VoltMeasConfMainWin
from PyQt5 import QtWidgets
from copy import deepcopy
import logging

class VoltMeasConfUi(QtWidgets.QMainWindow, Ui_VoltMeasConfMainWin):
    def __init__(self, main, default_dict):
        super(VoltMeasConfUi, self).__init__()
        self.setupUi(self)

        self.main = main

        self.doubleSpinBox_measVoltPulseLength_mu_s.valueChanged.connect(self.pulse_length)
        self.doubleSpinBox_measVoltTimeout_mu_s_set.valueChanged.connect(self.timeout)
        self.default_vals = deepcopy(default_dict)
        self.set_values_by_dict(self.default_vals)

        try:
            self.set_values_by_dict(self.default_vals)
        except Exception as e:
            logging.error('could not load the default values: ' + str(self.default_vals)
                          + ' to the gui.\n Exception is:' + str(e))
        self.show()

    def set_values_by_dict(self, meas_volt_dict):
        self.doubleSpinBox_measVoltPulseLength_mu_s.setValue(meas_volt_dict['measVoltPulseLength25ns'] * 25 / 1000)
        self.doubleSpinBox_measVoltTimeout_mu_s_set.setValue(meas_volt_dict['measVoltTimeout10ns'] / 100)

    def pulse_length(self, pulse_len_mu_s):
        pulse_len_25ns = int(round(pulse_len_mu_s / 25 * 1000))
        self.main.w_global_scanpars('measVoltPulseLength25ns', pulse_len_25ns)
        pulse_len_mu_s = pulse_len_25ns * 25 / 1000
        self.label_measVoltPulseLength_mu_s_set.setText('{0:0.3f}'.format(pulse_len_mu_s))

    def timeout(self, timeout_mu_s):
        timeout_10ns = int(round(timeout_mu_s * 100))
        self.main.w_global_scanpars('measVoltTimeout10ns', timeout_10ns)
        timeout_mu_s = timeout_10ns / 100
        self.label_measVoltTimeout_mu_s_set.setText('{0:0.3f}'.format(timeout_mu_s))
