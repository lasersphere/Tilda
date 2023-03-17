"""
Created on 

@author: simkaufm

Module Description: since the sweep trigger is exactly the same as the single hit trigger
 but only is active on first step, just use the existing ui.
"""
import logging
from Tilda.Interface.TriggerWidgets.SingleHitWidgUi import SingelHit
from Tilda.Driver.DataAcquisitionFpga.TriggerTypes import TriggerTypes as TiTs


class SweepUi(SingelHit):
    def __init__(self, trigger_dict):
        super(SweepUi, self).__init__(trigger_dict)
        self.set_type()
        logging.debug('SweepUi has type: %s' % self.type)

    def set_type(self):
        self.type = TiTs.sweep
        logging.debug('trigger type set to sweep')
