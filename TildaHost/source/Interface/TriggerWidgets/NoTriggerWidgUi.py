"""
Created on 

@author: simkaufm

Module Description:
"""

from Interface.TriggerWidgets.BaseTriggerWidg import BaseTriggerWidgUi
from Interface.TriggerWidgets.Ui_NoTrigger import Ui_no_trigger_widg
from Driver.DataAcquisitionFpga.TriggerTypes import TriggerTypes as TiTs


class NoTriggerWidg(BaseTriggerWidgUi, Ui_no_trigger_widg):
    def __init__(self, trigger_dict):
        BaseTriggerWidgUi.__init__(self, trigger_dict)
        self.setupUi(self)
        self.buffer_pars = {}

    def set_type(self):
        self.type = TiTs.no_trigger

