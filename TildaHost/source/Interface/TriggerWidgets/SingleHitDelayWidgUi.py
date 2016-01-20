"""
Created on 

@author: simkaufm

Module Description:
"""

from Interface.TriggerWidgets.BaseTriggerWidg import BaseSequencerWidgUi
from Interface.TriggerWidgets.Ui_SingleHitDelay import Ui_single_hit_delay_widg
from Driver.DataAcquisitionFpga.TriggerTypes import TriggerTypes as TiTs


class SingelHitDelay(BaseSequencerWidgUi, Ui_single_hit_delay_widg):
    def __init__(self, trigger_dict):
        BaseSequencerWidgUi.__init__(self, trigger_dict)
        self.trig_chan_list = ['Ext%s' % ch for ch in range(0, 8)] + ['Int%s' % ch for ch in range(0, 8)]
        self.comboBox_trigInputChan.addItems(self.trig_chan_list)

    def set_type(self):
        self.type = TiTs.single_hit_delay

    def connect_labels(self):
        self.setupUi(self)
        self.comboBox_trigInputChan.currentTextChanged.connect(self.trig_input)
        self.doubleSpinBox_trigDelay_mus.valueChanged.connect(self.trig_delay_set)

    def set_vals_by_dict(self):
        if self.buffer_pars.get('trigInputChan', -1) >= 0:
            if self.buffer_pars.get('trigInputChan') is not None:
                trig_ind = self.buffer_pars.get('trigInputChan')
                self.comboBox_trigInputChan.setCurrentIndex(trig_ind)
        if self.buffer_pars.get('trigDelay10ns', False):
            if self.buffer_pars.get('trigDelay10ns') is not None:
                delay = self.buffer_pars.get('trigDelay10ns')
                self.doubleSpinBox_trigDelay_mus.setValue(delay * (10 ** -2))

    def trig_input(self, trig_name):
        ind = self.comboBox_trigInputChan.currentIndex()
        self.buffer_pars['trigInputChan'] = ind
        self.label_trigInputChan_set.setText(trig_name + '  ## ' + str(ind))

    def trig_delay_set(self, val):
        self.buffer_pars['trigDelay10ns'] = int(val * (10 ** 2))
        self.label_trigDelay_mus_set.setText(str(val))


