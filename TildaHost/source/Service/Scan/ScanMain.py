"""

Created on '19.05.2015'

@author:'simkaufm'

"""

import time

from Driver.DataAcquisitionFpga.TimeResolvedSequencer import TimeResolvedSequencer as trs
from Driver.DataAcquisitionFpga.ContinousSequencer import ContinousSequencer as cs
from Driver.Heinzinger.Heinzinger import Heinzinger
import Driver.Heinzinger.HeinzingerCfg as hzCfg


class ScanMain():
    def __init__(self):
        self.sequencer = None
        self.scan_state = 'initialized'

        # self.heinz0 = Heinzinger(hzCfg.comportHeinzinger0)
        # self.heinz1 = Heinzinger(hzCfg.comportHeinzinger1)
        # self.heinz2 = Heinzinger(hzCfg.comportHeinzinger2)


    def start_measurement(self, scan_dict):
        self.prepare_measurement(scan_dict)
        self.scan_state = 'measuring'
        # self.setHeinzinger(scanpars)
        # if scanpars['measureOffset']:
        #     self.measureOffset(scanpars)
        # self.measureOneTrack(scanpars)

    def prepare_measurement(self, scan_dict):
        self.scan_state = 'setting up measurement'
        if self.sequencer is None:
            # dynamically load the chosen fpga here. if not loaded anyhow yet.
            pass

    def measureOneTrack(self, scanpars):
        #dont like this. its just forwarding isnt it?
        self.trs.measureTrack(scanpars)

    def setHeinzinger(self, scanpars):
        """
        function to set the desired Heinzinger to the Voltage that is needed.
        :param scanpars: dcitionary, containing all scanparameters
        :return: bool, True if success, False if fail within maxTries.
        """
        activeHeinzinger = getattr(self, 'heinz' + str(scanpars['postAccOffsetVoltControl']))
        setVolt = scanpars['postAccOffsetVolt']
        if setVolt*0.9 < activeHeinzinger.getVoltage() < setVolt*1.1:
            #Voltage already set, not needed to change it
            return True
        else:
            #set Voltage and compare it with desired Voltage.
            activeHeinzinger.setVoltage(setVolt)
            tries = 0
            maxTries = 10
            while not setVolt*0.9 < activeHeinzinger.getVoltage() < setVolt*1.1:
                time.sleep(0.1)
                tries += 1
                if tries > maxTries:
                    return False
            return True

    def measureOffset(self, scanpars):
        """
        Measure the Offset Voltage using a digital Multimeter. Hopefully the NI-4071
        will be implemented in the future.
        :param scanpars: dictionary, containing all scanparameters
        :return: bool, True if success
        """
        return True
