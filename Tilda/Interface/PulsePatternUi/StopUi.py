"""
Created on 19.01.2017

@author: simkaufm

Module Description:  Dialog for editing a channel item in the periodic input gui for teh pulsepattern gen
"""
import ast
from PyQt5 import QtWidgets

from Tilda.Interface.PulsePatternUi.Ui_Stop import Ui_StopUi


class StopUi(QtWidgets.QDialog, Ui_StopUi):
    def __init__(self, parent, trig_dict):
        super(StopUi, self).__init__(parent)
        self.setupUi(self)
        self.trig_dict = trig_dict

        self.load_dial_from_dict(trig_dict)

    def load_dial_from_dict(self, trig_dict):
        """ load all info in dict to the controls in the gui """
        self.lineEdit.setText(str(trig_dict['actCh'])[1:-1])

    def get_dict_from_gui(self):
        """ readout all values from the gui and return them """
        ret_trig_dict = {
            'type': 'stop',
            'actCh': ast.literal_eval('[%s]' % self.lineEdit.text())
        }
        return ret_trig_dict
