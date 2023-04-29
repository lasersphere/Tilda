"""
Created on 05.03.2022

@author: Patrick Mueller
"""

from PyQt5 import QtWidgets, QtCore

from Tilda.Interface.ArithmeticsUi.ArithmeticsUi import ArithmeticsUi
from Tilda.Interface.DopplerConfigUi.Ui_DopplerConfig import Ui_DopplerConfig
import Tilda.Application.Config as Cfg
from Tilda.PolliFit.Physics import wavenumber, wavelenFromFreq


def setup_doppler_config():
    options = Cfg._main_instance.local_options
    freq_dict, freq_arith = options.get_freq_settings()
    volt_dict, volt_arith = options.get_volt_settings()
    doppler = options.get_doppler_settings()
    doppler_config = dict(
        mass=float(doppler['mass']),
        charge=int(doppler['charge']),
        col=bool(doppler['col']),
        freq_dict=freq_dict,
        freq_arith=freq_arith,
        volt_dict=volt_dict,
        volt_arith=volt_arith,
        amplifier_slope=doppler['amplifier_slope'],
        amplifier_offset=doppler['amplifier_offset']
    )
    return doppler_config


def calc_arith(obs_dict, obs_arith):
    return eval(obs_arith, {'__builtins__': None}, obs_dict)


class DopplerConfigUi(QtWidgets.QDialog, Ui_DopplerConfig):
    close_signal = QtCore.pyqtSignal()

    def __init__(self, parent=None, doppler_config=None):
        super(DopplerConfigUi, self).__init__(parent=parent)
        self.setupUi(self)
        self.freq_win = None
        self.volt_win = None
        if doppler_config is None:
            doppler_config = setup_doppler_config()
        self.doppler_config = doppler_config
        self.freq_dict, self.freq_arith = self.doppler_config['freq_dict'], self.doppler_config['freq_arith']
        self.volt_dict, self.volt_arith = self.doppler_config['volt_dict'], self.doppler_config['volt_arith']
        self.save = True
        self.set_config_ui()
        
        self.b_config_frequency.clicked.connect(self.open_freq_win)
        self.b_config_voltage.clicked.connect(self.open_volt_win)
        self.b_ok.clicked.connect(self.close)
        self.b_cancel.clicked.connect(self.revert_and_close)

        self.show()

    def open_freq_win(self):
        preview = {'functions': [None, wavenumber, wavelenFromFreq],
                   'units': ['MHz', 'cm-1', 'nm'], 'decimals': [0, 2, 5]}
        self.freq_win = ArithmeticsUi(
            self, self.freq_dict, self.freq_arith, close_func=self.close_freq_win,
            obs_name='Laser Frequency', preview=preview)

    def open_volt_win(self):
        preview = {'functions': [None], 'units': ['V'], 'decimals': [3]}
        self.volt_win = ArithmeticsUi(self, self.volt_dict, self.volt_arith, close_func=self.close_volt_win,
                                      obs_name='Acceleration Voltage', preview=preview)

    def close_freq_win(self):
        if self.freq_win.val_accepted:
            self.freq_dict, self.freq_arith = self.freq_win.obs_dict, self.freq_win.obs_arith
            self.d_laser_frequency.setValue(calc_arith(self.freq_dict, self.freq_arith))
        self.freq_win = None

    def close_volt_win(self):
        if self.volt_win.val_accepted:
            self.volt_dict, self.volt_arith = self.volt_win.obs_dict, self.volt_win.obs_arith
            self.d_voltage.setValue(calc_arith(self.volt_dict, self.volt_arith))
        self.volt_win = None

    def set_config_ui(self):
        self.d_mass.setValue(self.doppler_config['mass'])
        self.s_charge.setValue(self.doppler_config['charge'])
        self.check_col.setChecked(self.doppler_config['col'])
        self.d_laser_frequency.setValue(calc_arith(self.freq_dict, self.freq_arith))
        self.d_voltage.setValue(calc_arith(self.volt_dict, self.volt_arith))
        self.d_slope.setValue(self.doppler_config['amplifier_slope'])
        self.d_offset.setValue(self.doppler_config['amplifier_offset'])

    def set_config_dict(self):
        self.doppler_config['mass'] = self.d_mass.value()
        self.doppler_config['charge'] = self.s_charge.value()
        self.doppler_config['col'] = self.check_col.isChecked()
        self.doppler_config['freq_dict'] = self.freq_dict
        self.doppler_config['freq_arith'] = self.freq_arith
        self.doppler_config['volt_dict'] = self.volt_dict
        self.doppler_config['volt_arith'] = self.volt_arith
        self.doppler_config['amplifier_slope'] = self.d_slope.value()
        self.doppler_config['amplifier_offset'] = self.d_offset.value()

    def revert_and_close(self):
        self.save = False
        self.close()

    def close(self):
        if self.save:
            self.set_config_dict()
        self.close_signal.emit()
        return super().close()
