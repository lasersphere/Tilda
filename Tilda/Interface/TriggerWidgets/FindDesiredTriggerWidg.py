"""
Created on 

@author: simkaufm

Module Description:
"""

from Tilda.Interface.TriggerWidgets.SingleHitDelayWidgUi import SingelHitDelay
from Tilda.Interface.TriggerWidgets.SingleHitWidgUi import SingelHit
from Tilda.Interface.TriggerWidgets.NoTriggerWidgUi import NoTriggerWidg
from Tilda.Interface.TriggerWidgets.SweepTriggerWidgUi import SweepUi
from Tilda.Interface.TriggerWidgets.SoftwareTriggerWidgUi import SoftwareTriggerUi
from Tilda.Driver.DataAcquisitionFpga.TriggerTypes import TriggerTypes as TiTs

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
    elif trig_type is TiTs.software:
        logging.debug('loading ' + str(trig_type.name))
        return SoftwareTriggerUi(trigger_dict)
    elif trig_type is TiTs.sweep:
        logging.debug('loading ' + str(trig_type.name))
        return SweepUi(trigger_dict)
