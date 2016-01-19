"""
Created on 

@author: simkaufm

Module Description:
"""

from Interface.SequencerWidgets.BaseSequencerWidgUi import BaseSequencerWidgUi
from Interface.SequencerWidgets.Ui_ContSeqWidg import Ui_ContSeqWidg
import Service.Scan.draftScanParameters as DftSc

from PyQt5 import QtWidgets


class ContSeqWidg(QtWidgets.QFrame, Ui_ContSeqWidg):

    def __init__(self, track_par):
        QtWidgets.QFrame.__init__(self)
        self.type = None
        self.set_type()
        self.key_list = getattr(DftSc, self.type + '_list')
        self.setupUi(self)

        self.buffer_pars = {key: val for key, val in track_par.items() if key in self.key_list}
        self.connect_labels()
        self.set_vals_by_dict()

    def get_seq_pars(self):
        return self.buffer_pars

    def set_type(self):
        self.type = 'cs'

    def connect_labels(self):
        self.doubleSpinBox_dwellTime_ms.valueChanged.connect(self.dwelltime_set)

    def set_vals_by_dict(self):
        if self.buffer_pars.get('dwellTime10ns', False):
            if self.buffer_pars.get('dwellTime10ns') is not None:
                self.doubleSpinBox_dwellTime_ms.setValue(
                    self.buffer_pars['dwellTime10ns'] * (10 ** -5))

    def dwelltime_set(self, val):
        """ this will write the doublespinbox value to the working dict and set the label """
        self.buffer_pars['dwellTime10ns'] = val * (10 ** 5)  # convert to units of 10ns
        self.label_dwellTime_ms_2.setText(str(val))

