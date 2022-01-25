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

        ''' get buffer parameter from main if available '''
        if Cfg._main_instance is not None:
            self.buffer_pars = deepcopy(Cfg._main_instance.cnt_pars)
        else:
            self.buffer_pars = deepcopy(dft.draftCntPars)
        if self.buffer_pars.get('trigger', None) is None:
            self.buffer_pars = deepcopy(dft.draftCntPars)
        logging.info('parameters are: %s ' % (self.buffer_pars))

        ''' setup gui and widgets'''
        self.trigger_widget = None  # trigger widget
        self.tg_widget = None   # time gate widget

        self.setupUi(self)

        ''' connect '''
        self.buttonBox.accepted.connect(self.ok)
        self.buttonBox.rejected.connect(self.close)
        self.lineEdit_act_pmts.textChanged.connect(self.set_act_pmts)   # which pmts are used
        self.spinBox_plotpoints.valueChanged.connect(self.set_datapoints)   # max of datapoints shown in one plot
        self.comboBox_triggerSelect.currentTextChanged.connect(self.trigger_select) # which trigger is used

        ''' default vals: '''
        self.lineEdit_act_pmts.setText('0, 1, 2, 3')
        self.spinBox_plotpoints.setValue(600)

        ''' trigger and timegate widgets '''
        self.tg_widget = NoTGWidg()
        self.verticalLayout_trigger.replaceWidget(self.widget_trigger_place_holder, self.trigger_widget)    # trigger
        self.verticalLayout_4.replaceWidget(self.widget_timeGate_place_holder, self.tg_widget)  # time gate
        self.comboBox_triggerSelect.addItems(tr.name for tr in TiTs)    # add trigger types to trigger menue
        self.comboBox_triggerSelect.setCurrentText(self.buffer_pars['trigger']['meas_trigger']['type'].name)    # set
        # trigger to last used trigger

        self.exec_()

    def ok(self):
        """
        if ok button is pressed, the buffer parameters are given to the main instance and the measurement is started
        """
        if Cfg._main_instance is not None:
            Cfg._main_instance.cnt_pars = deepcopy(self.buffer_pars)
        logging.info('confirmed cnt dict: %s ' % (self.buffer_pars))
        self.start = True

    def cancel(self):
        self.start = False

    def update_trigg_menue(self):
        """
        set shown values in the trigger menue (trigger delay) to matching values of the time gate menue
        """
        logging.info('updateding trigger menue')
        self.trigger_widget.set_vals_by_dict()

    def update_tg_menue(self):
        """
        set shown values in the time gate menue (mid tof) to matching values of the trigger menue
        """
        logging.info('updateing tg menue')
        self.tg_widget.set_vals_by_dict()

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
        :param trig_str: str, input from combobox defines used trigger
        :return:
        """

        ''' set buffer parameter to combobox input '''
        self.buffer_pars['trigger']['meas_trigger']['type'] = getattr(TiTs, trig_str)

        ''' adjust widgets accordingly'''
        self.verticalLayout_trigger.removeWidget(self.trigger_widget)   # replace triggerwidget
        if self.trigger_widget is not None:
            self.trigger_widget.setParent(None)
        self.trigger_widget = FindDesiredTriggerWidg.find_trigger_widget(self.buffer_pars.get('trigger',{}).get('meas_trigger', {}))
        self.verticalLayout_trigger.addWidget(self.trigger_widget)

        self.verticalLayout_4.removeWidget(self.tg_widget)  # replace time gate widget
        if self.tg_widget is not None:
            self.tg_widget.setParent(None)
        if self.buffer_pars['trigger']['meas_trigger']['type'] in [TiTs.single_hit_delay, TiTs.single_hit]:
            self.tg_widget = TGWidg(self.buffer_pars)
            self.verticalLayout_4.addWidget(self.tg_widget)
            ''' synchronize shown parameters in time gate and trigger widgets'''
            self.tg_widget.lineEdit_mid_tof.textChanged.connect(self.update_trigg_menue)
            self.tg_widget.lineEdit_gate_width.textChanged.connect(self.update_trigg_menue)
            self.trigger_widget.doubleSpinBox_trigDelay_mus.valueChanged.connect(self.update_tg_menue)
        else:   # no time gate widget needed for NoTrigger
            self.tg_widget = NoTGWidg()
            self.verticalLayout_4.addWidget(self.tg_widget)

