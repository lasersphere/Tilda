"""
Created on 

@author: simkaufm

Module Description: Module to combine all power supplies needed for the PostAcceleration
The selection which one is connected to the electrode is done via the FPGA (->postAccelerationOffsetControl)
"""

import logging
import time

from Driver.Heinzinger.Heinzinger import Heinzinger
import Driver.Heinzinger.HeinzingerCfg as HzCfg


class PostAccelerationMain:
    def __init__(self):
        self.active_power_supplies = {}

    def power_supply_init(self):
        """
        trying to initialize all Powersupplies
        """
        self.active_power_supplies = {}
        i = 0
        for com in [HzCfg.comportHeinzinger1, HzCfg.comportHeinzinger2, HzCfg.comportHeinzinger3]:
            i += 1
            if com > 0:
                name = 'Heinzinger' + str(i)
                try:
                    dev = Heinzinger(com, name)
                    if dev.idn != str(None):
                        self.active_power_supplies[name] = Heinzinger(com, name)
                except Exception as e:
                    logging.error('While initialising ' + name + ' on com port ' +
                                  str(com) + ' the following error occured: ' + str(e))
        logging.debug('active postAcceleration power supplies: ' + str(self.active_power_supplies))
        return self.active_power_supplies

    def status_of_power_supply(self, power_supply):
        """
        returns a dict containing the status of the power supply,
        keys are: name, programmedVoltage, voltageSetTime, readBackVolt, output
        """
        power_sup = self.active_power_supplies.get(power_supply, False)
        if power_sup:
            return power_sup.get_status()
        else:
            return None

    def set_voltage(self, power_supply, voltage):
        """
        will set the voltage to the desired powersupply
        """
        power_sup = self.active_power_supplies.get(power_supply, False)
        if power_sup:
            # don't touch the device, if the programmed voltage is the same as the desired one
            if power_sup.setVolt != voltage:
                # setting the voltage if not already done before
                power_sup.setVoltage(voltage)
            tries = 0
            maxTries = 10
            readback = power_sup.getVoltage()
            while not voltage * 0.95 < readback < voltage * 1.05 or (voltage < 0.1 and readback < 1):
                time.sleep(0.1)
                tries += 1
                readback = power_sup.getVoltage()
                if tries > maxTries:
                    logging.warning(str(power_sup) + ' readback is not within 10% of desired voltage,\n Readback is: ' +
                                    str(readback))
                    return readback
            logging.info(str(power_sup) +
                         'readback is: ' + str(readback) + ' V\n' +
                         'last set at: ' + power_sup.time_of_last_volt_set)
            return readback
        else:
            logging.debug(power_supply + ' is not active. Voltage can not be set.')
            return None
