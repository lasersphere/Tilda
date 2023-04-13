"""
Created on 05.03.2022

@author: Patrick Mueller
"""


from copy import deepcopy
from PyQt5 import QtWidgets, QtCore
from Tilda.PolliFit.Gui.Ui_ColAcolConfig import Ui_ColAcolConfig


class ColAcolConfigUi(QtWidgets.QWidget, Ui_ColAcolConfig):
    gate_signal = QtCore.pyqtSignal()

    def __init__(self, parameters, col_acol_config):
        super(ColAcolConfigUi, self).__init__()
        self.setupUi(self)
        self.c_parameter.addItems(parameters if parameters else ['center', 'x0'])

        self.parameters = parameters
        self.col_acol_config = None
        self.set_config(col_acol_config)
        self.old_col_acol_config = deepcopy(self.col_acol_config)

        self.c_rules.currentIndexChanged.connect(self.set_rule)
        self.c_parameter.currentIndexChanged.connect(self.set_parameter)
        self.s_iterate.valueChanged.connect(self.set_iterate)
        self.d_volt.valueChanged.connect(self.set_volt)
        self.d_mhz.valueChanged.connect(self.set_mhz)
        self.check_mc.stateChanged.connect(self.set_mc)
        self.s_mc.valueChanged.connect(self.set_mc_size)
        self.check_voltage.stateChanged.connect(self.set_save_voltage)
        self.check_show.stateChanged.connect(self.set_show_results)
        self.check_save.stateChanged.connect(self.set_save_results)
        self.line_file.textChanged.connect(self.set_file)

        self.b_ok.clicked.connect(self.close)
        self.b_cancel.clicked.connect(self.revert_and_close)

    def set_config(self, config):
        self.col_acol_config = config
        self.c_rules.setCurrentText(config['rule'])
        if config['iterate'] in self.parameters:
            self.c_parameter.setCurrentText(config['parameter'])
        else:
            if 'center' in self.parameters:
                self.c_parameter.setCurrentText('center')
            else:
                self.c_parameter.setCurrentText('x0')
            self.set_parameter()
        self.s_iterate.setValue(config['iterate'])
        self.d_volt.setValue(config['volt'])
        self.d_mhz.setValue(config['mhz'])
        self.check_mc.setChecked(config['mc'])
        self.s_mc.setValue(config['mc_size'])
        self.check_voltage.setChecked(config['save_voltage'])
        self.check_show.setChecked(config['show_results'])
        self.check_save.setChecked(config['save_results'])
        self.line_file.setText(config['file'])

    def set_rule(self):
        self.col_acol_config['rule'] = self.c_rules.currentText()

    def set_parameter(self):
        self.col_acol_config['parameter'] = self.c_parameter.currentText()

    def set_iterate(self):
        self.col_acol_config['iterate'] = self.s_iterate.value()

    def set_volt(self):
        self.col_acol_config['volt'] = self.d_volt.value()

    def set_mhz(self):
        self.col_acol_config['mhz'] = self.d_mhz.value()

    def set_mc(self):
        self.col_acol_config['mc'] = self.check_mc.isChecked()

    def set_mc_size(self):
        self.col_acol_config['mc_size'] = self.s_mc.value()

    def set_save_voltage(self):
        self.col_acol_config['save_voltage'] = self.check_voltage.isChecked()

    def set_show_results(self):
        self.col_acol_config['show_results'] = self.check_show.isChecked()

    def set_save_results(self):
        self.col_acol_config['save_results'] = self.check_save.isChecked()

    def set_file(self):
        self.col_acol_config['file'] = self.line_file.text()

    def close(self):
        self.gate_signal.emit()
        super().close()

    def revert_and_close(self):
        self.col_acol_config = self.old_col_acol_config
        self.gate_signal.emit()
        super().close()
