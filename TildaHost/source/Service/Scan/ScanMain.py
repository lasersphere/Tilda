"""

Created on '19.05.2015'

@author:'simkaufm'

"""

import time

from Driver.DataAcquisitionFpga.TimeResolvedSequencer import TimeResolvedSequencer
from Driver.Heinzinger.Heinzinger import Heinzinger
import Driver.Heinzinger.HeinzingerCfg as hzCfg

class ScanMain():
    def __init__(self):
        self.trs = TimeResolvedSequencer()
        # self.heinz0 = Heinzinger(hzCfg.comportHeinzinger0)
        # self.heinz1 = Heinzinger(hzCfg.comportHeinzinger1)
        # self.heinz2 = Heinzinger(hzCfg.comportHeinzinger2)


    def startMeasurement(self, scanpars):
        self.setHeinzinger(scanpars)
        if scanpars['measureOffset']:
            self.measureOffset(scanpars)
        self.measureOneTrack(scanpars)



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
        setVolt = scanpars['heinzingerOffsetVolt']
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
