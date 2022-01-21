"""

Created on '25.11.2015'

@author:'simkaufm'

Description: Dialog to control the Simple Counter while this is running.
"""

from PyQt5 import QtWidgets, QtGui

import logging
from Interface.SimpleCounter.Ui_Simp_Count_Dial import Ui_Dialog_simpleCounterControl
import Application.Config as Cfg
import ast
from copy import deepcopy
from Driver.DataAcquisitionFpga.TriggerTypes import TriggerTypes as TiTs
import Interface.TriggerWidgets.FindDesiredTriggerWidg as FindDesiredTriggerWidg
from Interface.SimpleCounter.NoTGWidgUi import NoTGWidg
from Interface.SimpleCounter.TGWidgUi import TGWidg
import Service.SimpleCounter.draftCntParameters as dft


class SimpleCounterDialogUi(QtWidgets.QDialog, Ui_Dialog_simpleCounterControl):
    def __init__(self):
        super(SimpleCounterDialogUi, self).__init__()

        self.start = False
        self.act_pmts = []
        self.datapoints = 0
        if Cfg._main_instance is not None:
            print(Cfg._main_instance.cnt_pars)
            self.buffer_pars = deepcopy(Cfg._main_instance.cnt_pars)
        else:
            self.buffer_pars = deepcopy(dft.draftCntPars)
        if self.buffer_pars.get('trigger', None) is None:
            self.buffer_pars = deepcopy(dft.draftCntPars)

        logging.info('parameters are: %s ' % (self.buffer_pars))
        #self.buffer_pars = {'trigger': {'meas_trigger': {'type': TiTs.no_trigger}}, 'tg': {'gate_width': 0, 'mid_tof': 0}}
        self.trigger_widget = None
        self.tg_widget = None

        self.setupUi(self)

        self.buttonBox.accepted.connect(self.ok)
        self.buttonBox.rejected.connect(self.close)
        self.lineEdit_act_pmts.textChanged.connect(self.set_act_pmts)
        self.spinBox_plotpoints.valueChanged.connect(self.set_datapoints)
        self.comboBox_triggerSelect.currentTextChanged.connect(self.trigger_select)

        ''' default vals: '''
        self.lineEdit_act_pmts.setText('0, 1, 2, 3')
        self.spinBox_plotpoints.setValue(600)

        trig_type = TiTs.no_trigger
        self.comboBox_triggerSelect.setCurrentText(trig_type.name)
        self.trigger_widget = FindDesiredTriggerWidg.find_trigger_widget(self.buffer_pars.get('trigger',{}).get('meas_trigger',{}))
        self.verticalLayout_trigger.replaceWidget(self.widget_trigger_place_holder, self.trigger_widget)
        self.tg_widget = NoTGWidg()
        self.verticalLayout_4.replaceWidget(self.widget_timeGate_place_holder, self.tg_widget)
        self.comboBox_triggerSelect.addItems(tr.name for tr in TiTs)

        self.exec_()

    def ok(self):
        if Cfg._main_instance is not None:
            Cfg._main_instance.cnt_pars = deepcopy(self.buffer_pars)
        logging.info('confirmed cnt dict: %s ' % (self.buffer_pars))
        self.start = True

    def cancel(self):
        self.start = False

    def set_act_pmts(self, lis_str):
        """ convert input line to list and set the label """
        if type(lis_str) == str:
            try:
                lis_str = str('[' + lis_str + ']')
                lis = ast.literal_eval(lis_str)
            except Exception as e:
                lis = []
        self.label_act_pmts_set.setText(str(lis))
        self.act_pmts = lis

    def set_datapoints(self, val):
        self.datapoints = val
        self.label_plotpoints_set.setText(str(val))

    def trigger_select(self, trig_str):
        """
        finds the desired trigger widget and sets it into self.trigger_widget
        """
        self.buffer_pars['trigger']['meas_trigger']['type'] = getattr(TiTs, trig_str)
        self.verticalLayout_trigger.removeWidget(self.trigger_widget)

        if self.trigger_widget is not None:
            self.trigger_widget.setParent(None)
        self.trigger_widget = FindDesiredTriggerWidg.find_trigger_widget(self.buffer_pars.get('trigger',{}).get('meas_trigger', {}))
        self.verticalLayout_trigger.addWidget(self.trigger_widget)

        self.verticalLayout_4.removeWidget(self.tg_widget)
        if self.tg_widget is not None:
            self.tg_widget.setParent(None)
        if self.buffer_pars['trigger']['meas_trigger']['type'] in [TiTs.single_hit_delay, TiTs.single_hit]:
            self.tg_widget = TGWidg(self.buffer_pars.get('trigger', {}).get('tg', {}))
            self.verticalLayout_4.addWidget(self.tg_widget)
        else:
            self.tg_widget = NoTGWidg()
            self.verticalLayout_4.addWidget(self.tg_widget)

