"""
Created on 

@author: simkaufm

Module Description:
"""

from Interface.SequencerWidgets.BaseSequencerWidg import BaseSequencerWidgUi
from Interface.SequencerWidgets.Ui_TRSWidg import Ui_TRSWidg
import TildaTools as TiTs

import ast
import logging


class TRSWidg(BaseSequencerWidgUi, Ui_TRSWidg):
    def __init__(self, track_dict, main_gui):
        BaseSequencerWidgUi.__init__(self, track_dict)
        self.main_gui = main_gui

    def set_type(self):
        self.type = 'trs'

    def connect_labels(self):
        self.setupUi(self)
        self.spinBox_nOfBins.valueChanged.connect(self.n_of_bins_set)
        self.spinBox_softBinWidth.valueChanged.connect(self.software_bin_width_set)
        self.spinBox_nOfBunches.valueChanged.connect(self.n_of_bunches_set)
        self.lineEdit.textChanged.connect(self.softw_gates_set)
        self.doubleSpinBox_gate_width.valueChanged.connect(self.gate_width_set)
        self.doubleSpinBox_mid_tof.valueChanged.connect(self.mid_tof_set)

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

    def mid_tof_set(self, val):
        self.label_mid_tof_set.setText('%.2f us' % val)
        self.calc_softw_gates_from_gui()

    def gate_width_set(self, val):
        self.label_gate_width_set.setText('%.2f us' % val)
        self.calc_softw_gates_from_gui()

    def softw_gates_set(self, text):
        try:
            inp = ast.literal_eval(text)
            if isinstance(inp, list):
                self.label_softwGates_set.setText(str(inp).replace('],', '],\n'))
            else:
                logging.debug('input is not a list')
        except Exception as e:
            logging.debug('you typed something invalid: ' + str(e))
        self.calc_softw_gates_from_gui()

    def calc_softw_gates_from_gui(self):
        delay_list = self.lineEdit.text()
        try:
            delay_list = ast.literal_eval(delay_list)
            if isinstance(delay_list, list):
                self.label_softwGates_set.setText(str(delay_list).replace('],', '],\n'))
            else:
                logging.debug('input is not a list')
        except Exception as e:
            logging.debug('you typed something invalid: ' + str(e))
        softw_gates = TiTs.calc_soft_gates_from_db_pars(
            self.doubleSpinBox_gate_width.value(),
            delay_list,
            self.doubleSpinBox_mid_tof.value(),
            voltage_gates=[-10.0, 10.0]
        )
        logging.debug('software gates are: ' + str(softw_gates))
        self.buffer_pars['softwGates'] = softw_gates

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
                run_gates_width, del_list, iso_mid_tof = TiTs.calc_db_pars_from_software_gate(
                    self.buffer_pars.get('softwGates'))
                logging.debug('software gates from dict: ' + str(self.buffer_pars.get('softwGates')))
                logging.debug('width: %.2f, del_list: %s, mid_tof: %.3f' % (run_gates_width, del_list, iso_mid_tof))
                self.doubleSpinBox_gate_width.setValue(run_gates_width)
                self.doubleSpinBox_mid_tof.setValue(iso_mid_tof)
                self.lineEdit.setText(str(del_list))
                self.lineEdit.textChanged.emit(str(del_list))
        else:
            lis = [[None]] * len(self.buffer_pars.get('activePmtList', []))
            self.lineEdit.setText(str(lis))
