"""
Created on 

@author: simkaufm

Module Description: Same as NoTrigger, so use existing
"""

from Interface.TriggerWidgets.NoTriggerWidgUi import NoTriggerWidg
from Driver.DataAcquisitionFpga.TriggerTypes import TriggerTypes as TiTs


class SoftwareTriggerUi(NoTriggerWidg):
    def __init__(self, trigger_dict):
        NoTriggerWidg.__init__(self, trigger_dict)
        self.set_type()

    def set_type(self):
        self.type = TiTs.software
