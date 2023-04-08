"""
Created on 23/04/2016

@author: simkaufm

Module Description:
"""

from Tilda.Interface.TriggerWidgets.BaseTriggerWidg import BaseTriggerWidgUi
from Tilda.Interface.TriggerWidgets.Ui_SingleHit import Ui_single_hit_widg
from Tilda.Driver.DataAcquisitionFpga.TriggerTypes import TriggerTypes as TiTs


class SingelHit(BaseTriggerWidgUi, Ui_single_hit_widg):
    def __init__(self, trigger_dict):
        BaseTriggerWidgUi.__init__(self, trigger_dict)
        self.comboBox_trigInputChan.blockSignals(True)
        self.trig_chan_list = ['Ext%s' % ch for ch in range(0, 8)] + ['Int%s' % ch for ch in range(0, 8)]
        self.comboBox_trigInputChan.addItems(self.trig_chan_list)
        self.comboBox_trigInputChan.blockSignals(False)
        self.comboBox_trigger_edge.blockSignals(True)
        self.trigger_edges = ['either', 'rising', 'falling']
        self.comboBox_trigger_edge.addItems(self.trigger_edges)
        self.comboBox_trigger_edge.blockSignals(False)

        self.set_vals_by_dict()

    def set_type(self):
        self.type = TiTs.single_hit

    def connect_labels(self):
        self.setupUi(self)
        self.comboBox_trigInputChan.currentIndexChanged.connect(self.trig_input)
        self.comboBox_trigger_edge.currentTextChanged.connect(self.trig_edge_changed)

    def set_vals_by_dict(self):
        trig_ind = self.buffer_pars.get('trigInputChan', 0)
        self.comboBox_trigInputChan.setCurrentIndex(trig_ind)
        self.trig_input(trig_ind)
        trig_edge = self.buffer_pars.get('trigEdge', 'rising')
        self.comboBox_trigger_edge.setCurrentText(trig_edge)

    def trig_input(self, ind):
        trig_name = self.comboBox_trigInputChan.currentText()
        self.buffer_pars['trigInputChan'] = ind
        self.label_trigInputChan_set.setText(trig_name + '  ## ' + str(ind))

    def trig_edge_changed(self, edge_str):
        self.buffer_pars['trigEdge'] = edge_str
        self.label_selected_trigger_edge.setText(edge_str)


