"""
Created on 19.01.2017

@author: simkaufm

Module Description:  Dialog for editing a channel item in the periodic input gui for teh pulsepattern gen
"""
import ast
from PyQt5 import QtWidgets

from Interface.PulsePatternUi.Ui_Trigger import Ui_TriggerUi


class TriggerUi(QtWidgets.QDialog, Ui_TriggerUi):
    def __init__(self, parent, trig_dict):
        super(TriggerUi, self).__init__(parent)
        self.setupUi(self)
        self.trig_dict = trig_dict

        self.load_dial_from_dict(trig_dict)

    def load_dial_from_dict(self, trig_dict):
        """ load all info in dict to the controls in the gui """
        self.lineEdit_trig_name.setText(trig_dict.get('trigName', 'NoName'))
        self.lineEdit_trig_channels.setText(str(trig_dict.get('trigChannels', []))[1:-1])
        self.lineEdit_act_ch.setText(str(trig_dict.get('actCh', []))[1:-1])

    def get_dict_from_gui(self):
        """ readout all values from the gui and return them """
        ret_trig_dict = {
            'type': 'trig',
            'trigName': self.lineEdit_trig_name.text(),
            'trigChannels': ast.literal_eval('[%s]' % self.lineEdit_trig_channels.text()),
            'actCh': ast.literal_eval('[%s]' % self.lineEdit_act_ch.text())
        }
        return ret_trig_dict
