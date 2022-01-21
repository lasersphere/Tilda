"""
Created on 2021-11-30

@author: lrenth

Module Description:
"""

from Interface.SimpleCounter.Ui_timeGating import Ui_timeGating
from PyQt5 import QtWidgets


class TGWidg(QtWidgets.QFrame, Ui_timeGating):
    def __init__(self, tg_dict):
        QtWidgets.QFrame.__init__(self)
        self.setupUi(self)
        if tg_dict is None:
            tg_dict = {}
        self.buffer_pars = tg_dict

        self.lineEdit_mid_tof.textChanged.connect(self.mid_tof_set)
        self.lineEdit_gate_width.textChanged.connect(self.gate_width_set)

        self.set_vals_by_dict()

    def set_vals_by_dict(self):
        #hier Einträge setzten
        try:
            gate_width=self.buffer_pars['nOfBins']/100
        except:
            gate_width = 10
            self.buffer_pars['nOfBins'] = gate_width * 100
        try:
            mid_tof = self.buffer_pars['trigger']['meas_trigger']['trigDelay10ns']/100 + gate_width/2
        except:
            mid_tof = gate_width/2
            self.buffer_pars['trigger']['meas_trigger']['trigDelay10ns'] = str(gate_width / 2 * 100)
        gate_width = str(gate_width)
        mid_tof = str(mid_tof)
        self.label_mid_tof.setText(mid_tof)
        self.label_gate_width.setText(gate_width)

    def mid_tof_set(self, value):
        self.buffer_pars['trigger']['meas_trigger']['trigDelay10ns'] = int(value) * 100 - int(self.buffer_pars['nOfBins']) / 2

    def gate_width_set(self, value):
        self.buffer_pars['nOfBins'] = int(value) * 100

    def get_tg_pars(self):  #TODO: irgendwo muss das aufgerufen werden, damit die buffer pars an den simplecounter gehen
        return self.buffer_pars
