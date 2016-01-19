"""
Created on 

@author: simkaufm

Module Description:
"""

from Interface.SequencerWidgets.BaseSequencerWidg import BaseSequencerWidgUi
from Interface.SequencerWidgets.Ui_ContSeqWidg import Ui_ContSeqWidg


class ContSeqWidg(BaseSequencerWidgUi, Ui_ContSeqWidg):
    def __init__(self, track_dict):
        BaseSequencerWidgUi.__init__(self, track_dict)

    def set_type(self):
        self.type = 'cs'

    def connect_labels(self):
        self.setupUi(self)
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

