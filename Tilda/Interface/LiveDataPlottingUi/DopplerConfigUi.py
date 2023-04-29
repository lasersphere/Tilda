"""
Created on 05.03.2022

@author: Patrick Mueller
"""


from copy import deepcopy
from PyQt5 import QtWidgets, QtGui, QtCore
from Tilda.Interface.LiveDataPlottingUi.Ui_DopplerConfig import Ui_DopplerConfig


class DopplerConfigUi(QtWidgets.QWidget, Ui_DopplerConfig):
    close_signal = QtCore.pyqtSignal()

    def __init__(self, doppler_config):
        super(DopplerConfigUi, self).__init__()
        self.setupUi(self)
        self.doppler_config = doppler_config
        self.save = True
        self.set_config_ui()

        self.line_freq_mult.setValidator(QtGui.QDoubleValidator())

        self.b_ok.clicked.connect(self.close)
        self.b_cancel.clicked.connect(self.revert_and_close)

    def close(self):
        if self.save:
            self.set_config_dict()
        self.close_signal.emit()
        return super().close()

    def set_config_ui(self):
        self.d_mass.setValue(self.doppler_config['mass'])
        self.s_charge.setValue(self.doppler_config['charge'])
        self.check_col.setChecked(self.doppler_config['col'])
        self.d_laser_frequency.setValue(self.doppler_config['laser_frequency'])
        self.line_freq_mult.setText(str(self.doppler_config['freq_mult']))
        self.d_voltage.setValue(self.doppler_config['voltage'])
        self.d_divider_ratio.setValue(self.doppler_config['divider_ratio'])
        self.d_slope.setValue(self.doppler_config['slope'])
        self.d_offset.setValue(self.doppler_config['offset'])

    def set_config_dict(self):
        self.doppler_config['mass'] = self.d_mass.value()
        self.doppler_config['charge'] = self.s_charge.value()
        self.doppler_config['col'] = self.check_col.isChecked()
        self.doppler_config['laser_frequency'] = self.d_laser_frequency.value()
        self.doppler_config['freq_mult'] = float(self.line_freq_mult.text())
        self.doppler_config['voltage'] = self.d_voltage.value()
        self.doppler_config['divider_ratio'] = self.d_divider_ratio.value()
        self.doppler_config['slope'] = self.d_slope.value()
        self.doppler_config['offset'] = self.d_offset.value()

    def revert_and_close(self):
        self.save = False
        self.close()
