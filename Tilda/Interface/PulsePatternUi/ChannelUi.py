"""
Created on 19.01.2017

@author: simkaufm

Module Description:  Dialog for editing a channel item in the periodic input gui for teh pulsepattern gen
"""
import ast
from PyQt5 import QtWidgets

from Tilda.Interface.PulsePatternUi.Ui_Channel import Ui_ChannelUi


class ChannelUi(QtWidgets.QDialog, Ui_ChannelUi):
    def __init__(self, parent, ch_dict, used_outputs):
        super(ChannelUi, self).__init__(parent)
        self.setupUi(self)
        self.channel_dict = ch_dict

        self.output_channels = ['DO%s' % i for i in range(0, 32)]
        self.output_channels = [ch for ch in self.output_channels if ch not in used_outputs]

        self.comboBox_out_ch.addItems(self.output_channels)
        self.comboBox_inverted.addItems(['False', 'True'])

        self.load_dial_from_dict(ch_dict)

    def load_dial_from_dict(self, ch_dict):
        """ load all info in dict to the controls in the gui """
        self.lineEdit_chan_name.setText(ch_dict.get('chName', ''))
        self.comboBox_out_ch.setCurrentText(ch_dict.get('outCh', ''))
        self.spinBox_num_pulses.setValue(ch_dict.get('numOfPulses', 1))
        self.doubleSpinBox_pulse_width_us.setValue(ch_dict.get('widthUs', 1.0))
        self.doubleSpinBox_delay_us.setValue(ch_dict.get('delayUs', 1.0))
        self.comboBox_inverted.setCurrentText(str(ch_dict.get('inverted', False)))

    def get_dict_from_gui(self):
        """ readout all values from the gui and return them """
        ret_ch_dict = {
            'type': 'ch',
            'chName': self.lineEdit_chan_name.text(),
            'outCh': self.comboBox_out_ch.currentText(),
            'numOfPulses': self.spinBox_num_pulses.value(),
            'widthUs': self.doubleSpinBox_pulse_width_us.value(),
            'delayUs': self.doubleSpinBox_delay_us.value(),
            'inverted': ast.literal_eval(self.comboBox_inverted.currentText())
        }
        return ret_ch_dict
