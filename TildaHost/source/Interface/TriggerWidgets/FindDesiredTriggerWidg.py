"""
Created on 

@author: simkaufm

Module Description:
"""

from Interface.TriggerWidgets.SingleHitDelayWidgUi import SingelHitDelay
from Interface.TriggerWidgets.SingleHitWidgUi import SingelHit
from Interface.TriggerWidgets.NoTriggerWidgUi import NoTriggerWidg
from Driver.DataAcquisitionFpga.TriggerTypes import TriggerTypes as TiTs

import logging


def find_trigger_widget(trigger_dict):
    trig_type = trigger_dict.get('type', TiTs.no_trigger)
    logging.debug('trigger_dict is: ' + str(trigger_dict))
    if trig_type is TiTs.no_trigger:
        logging.debug('loading ' + str(trig_type.name))
        return NoTriggerWidg(trigger_dict)
    elif trig_type is TiTs.single_hit_delay:
        logging.debug('loading ' + str(trig_type.name))
        return SingelHitDelay(trigger_dict)
    elif trig_type is TiTs.single_hit:
        logging.debug('loading ' + str(trig_type.name))
        return SingelHit(trigger_dict)


