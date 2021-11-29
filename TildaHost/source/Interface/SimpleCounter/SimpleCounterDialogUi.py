"""

Created on '25.11.2015'

@author:'simkaufm'

Description: Dialog to control the Simple Counter while this is running.
"""

from PyQt5 import QtWidgets, QtGui

from Interface.SimpleCounter.Ui_Simp_Count_Dial import Ui_Dialog_simpleCounterControl
import Application.Config as Cfg
import ast
from Driver.DataAcquisitionFpga.TriggerTypes import TriggerTypes as TiTs
import Interface.TriggerWidgets.FindDesiredTriggerWidg as FindDesiredTriggerWidg
#import Interface.SimpleCounter.no_TG_widget as no_tg_widget


class SimpleCounterDialogUi(QtWidgets.QDialog, Ui_Dialog_simpleCounterControl):
    def __init__(self):
        super(SimpleCounterDialogUi, self).__init__()

        self.start = False
        self.act_pmts = []
        self.datapoints = 0
        self.buffer_pars = {'type': TiTs.no_trigger}
        self.trigger_widget = None

        self.setupUi(self)

        self.buttonBox.accepted.connect(self.ok)
        self.buttonBox.rejected.connect(self.close)
        self.lineEdit_act_pmts.textChanged.connect(self.set_act_pmts)
        self.spinBox_plotpoints.valueChanged.connect(self.set_datapoints)
        self.comboBox_triggerSelect.currentTextChanged.connect(self.trigger_select)

        ''' default vals: '''
        self.lineEdit_act_pmts.setText('0, 1, 2, 3')
        self.spinBox_plotpoints.setValue(600)
        self.comboBox_triggerSelect.addItems(tr.name for tr in TiTs)
        trig_type = TiTs.no_trigger
        self.comboBox_triggerSelect.setCurrentText(trig_type.name)
        self.trigger_widget = FindDesiredTriggerWidg.find_trigger_widget(self.buffer_pars)
        self.verticalLayout_trigger.replaceWidget(self.widget_trigger_place_holder, self.trigger_widget)
        #self.verticalLayout.replaceWidget(self.widget_timeGate_place_holder,)

        self.exec_()

    def ok(self):
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
        self.buffer_pars['type'] = getattr(TiTs, trig_str)
        self.verticalLayout_trigger.removeWidget(self.trigger_widget)

        if self.trigger_widget is not None:
            self.trigger_widget.setParent(None)
        self.trigger_widget = FindDesiredTriggerWidg.find_trigger_widget(self.buffer_pars)
        self.verticalLayout_trigger.addWidget(self.trigger_widget)


