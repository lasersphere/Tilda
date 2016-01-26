"""
Created on 

@author: simkaufm

Module Description:
"""

from Interface.SequencerWidgets.BaseSequencerWidg import BaseSequencerWidgUi
from Interface.SequencerWidgets.Ui_TRSWidg import Ui_TRSWidg

import ast
import logging


class TRSWidg(BaseSequencerWidgUi, Ui_TRSWidg):
    def __init__(self, track_dict):
        BaseSequencerWidgUi.__init__(self, track_dict)

    def set_type(self):
        self.type = 'trs'

    def connect_labels(self):
        self.setupUi(self)
        self.spinBox_nOfBins.valueChanged.connect(self.n_of_bins_set)
        self.spinBox_nOfBunches.valueChanged.connect(self.n_of_bunches_set)
        self.lineEdit.textChanged.connect(self.softw_gates_set)

    def n_of_bins_set(self, val):
        self.buffer_pars['nOfBins'] = val
        self.label_nOfBins_set.setText(str(val))

    def n_of_bunches_set(self, val):
        self.buffer_pars['nOfBunches'] = val
        self.label_nOfBunches_set.setText(str(val))

    def softw_gates_set(self, text):
        try:
            print(self.buffer_pars)
            inp = ast.literal_eval(text)
            if isinstance(inp, list):
                self.label_softwGates_set.setText(str(inp).replace('],', '],\n'))
                self.buffer_pars['softwGates'] = inp
            else:
                logging.debug('input is not a list')
        except Exception as e:
            logging.debug('you typed something invalid: ' + str(e))

    def set_vals_by_dict(self):
        if self.buffer_pars.get('nOfBins', False):
            if self.buffer_pars.get('nOfBins') is not None:
                self.spinBox_nOfBins.setValue(self.buffer_pars.get('nOfBins'))
        if self.buffer_pars.get('nOfBunches', False):
            if self.buffer_pars.get('nOfBunches') is not None:
                self.spinBox_nOfBunches.setValue(self.buffer_pars.get('nOfBunches'))
        if self.buffer_pars.get('softwGates', False):
            if self.buffer_pars.get('softwGates') is not None:
                self.lineEdit.setText(str(self.buffer_pars.get('softwGates')))
        else:
            lis = [[None]] * len(self.buffer_pars.get('activePmtList', []))
            self.lineEdit.setText(str(lis))
