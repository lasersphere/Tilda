"""
Created on 

@author: simkaufm

Module Description:
"""

from Interface.TriggerWidgets.SingleHitDelayWidgUi import SingelHitDelay
from Interface.TriggerWidgets.NoTriggerWidgUi import NoTriggerWidg
from Driver.DataAcquisitionFpga.TriggerTypes import TriggerTypes as TiTs


def find_trigger_widget(trig_type, trigger_dict):
    if trig_type is TiTs.no_trigger:
        print('loading ', trig_type.name)
        return NoTriggerWidg(trigger_dict)
    elif trig_type is TiTs.single_hit_delay:
        print('loading ', trig_type.name)
        return SingelHitDelay(trigger_dict)

