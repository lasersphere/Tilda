"""
Created on 2021-11-30

@author: lrenth

Module Description: Widget to set time gate parameters for SimpleCounter
"""

import logging
from Interface.SimpleCounter.Ui_timeGating import Ui_timeGating
from PyQt5 import QtWidgets


class TGWidg(QtWidgets.QFrame, Ui_timeGating):
    def __init__(self, tg_dict):
        QtWidgets.QFrame.__init__(self)
        self.setupUi(self)

        '''get buffer parameters from parent '''
        if tg_dict is None:
            tg_dict = {}
        self.buffer_pars = tg_dict

        ''' connect '''
        self.lineEdit_mid_tof.textChanged.connect(self.mid_tof_set)
        self.lineEdit_gate_width.textChanged.connect(self.gate_width_set)

        ''' show correct values in the gui '''
        self.set_vals_by_dict()

    def set_vals_by_dict(self):
        """
        takes values from buffer_pars and sets lables and lineEdits in gui accordingly
        """

        ''' calculate gate width '''
        try:
            gate_width=self.buffer_pars['nOfBins']/100
        except:
            gate_width = 10
            self.buffer_pars['nOfBins'] = gate_width * 100

        ''' calculate mid tof'''
        try:
            mid_tof = self.buffer_pars['trigger']['meas_trigger']['trigDelay10ns']/100 + gate_width/2
        except:
            mid_tof = gate_width/2
            self.buffer_pars['trigger']['meas_trigger']['trigDelay10ns'] = str(gate_width / 2 * 100)

        ''' update views '''
        self.label_mid_tof.setText(str(mid_tof))
        self.label_gate_width.setText(str(gate_width))

    def mid_tof_set(self, value):
        """
        writes user input for mid_tof in buffer_pars
        :param value: int: mid tof in micro secs
        """

        ''' calculate trigger delay in 10ns '''
        try:
            self.buffer_pars['trigger']['meas_trigger']['trigDelay10ns'] = int(value) * 100 - int(self.buffer_pars['nOfBins']) / 2
            self.set_vals_by_dict()
        except:
            logging.info('Wrong input for ToF')

    def gate_width_set(self, value):
        """
        writes user input for gate_width in buffer pars
        :param value: int: gate width in micro secs
        """

        ''' calculate current mid tof from buffer_pars '''
        mid_tof = (int(self.buffer_pars['trigger']['meas_trigger']['trigDelay10ns']) + int(self.buffer_pars['nOfBins']) / 2) /100

        ''' calculate time width in nOfBins (10ns) '''
        self.buffer_pars['nOfBins'] = int(value) * 100

        ''' calculate new trigger delay in 10ns '''
        self.buffer_pars['trigger']['meas_trigger']['trigDelay10ns'] = (mid_tof - int(value) / 2) * 100

        self.set_vals_by_dict()
