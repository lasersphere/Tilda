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
        self.spinBox_softBinWidth.valueChanged.connect(self.software_bin_width_set)
        self.spinBox_nOfBunches.valueChanged.connect(self.n_of_bunches_set)
        self.lineEdit.textChanged.connect(self.softw_gates_set)

    def n_of_bins_set(self, val):
        self.buffer_pars['nOfBins'] = val
        time_in_mus = val / 100
        self.label_nOfBins_set.setText(str(val) + '| %.2f µs' % time_in_mus)

    def software_bin_width_set(self, val):
        val = val // 10 * 10
        self.blockSignals(True)
        self.spinBox_softBinWidth.setValue(val)
        self.buffer_pars['softBinWidth_ns'] = val
        self.label_softBinWidth_set.setText(str(val))
        self.blockSignals(False)

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
                self.spinBox_nOfBins.valueChanged.emit(self.buffer_pars.get('nOfBins'))
        if self.buffer_pars.get('softBinWidth_ns', False):
            if self.buffer_pars.get('softBinWidth_ns') is not None:
                self.spinBox_softBinWidth.valueChanged.emit(self.buffer_pars.get('softBinWidth_ns'))
        if self.buffer_pars.get('nOfBunches', False):
            if self.buffer_pars.get('nOfBunches') is not None:
                self.spinBox_nOfBunches.setValue(self.buffer_pars.get('nOfBunches'))
                self.spinBox_nOfBunches.valueChanged.emit(self.buffer_pars.get('nOfBunches'))
        if self.buffer_pars.get('softwGates', False):
            if self.buffer_pars.get('softwGates') is not None:
                self.lineEdit.setText(str(self.buffer_pars.get('softwGates')))
                self.lineEdit.textChanged.emit(str(self.buffer_pars.get('softwGates')))
        else:
            lis = [[None]] * len(self.buffer_pars.get('activePmtList', []))
            self.lineEdit.setText(str(lis))
