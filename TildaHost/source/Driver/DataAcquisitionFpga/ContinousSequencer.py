"""

Created on '09.07.2015'

@author:'simkaufm'

"""

from Driver.DataAcquisitionFpga.SequencerCommon import Sequencer
from Driver.DataAcquisitionFpga.MeasureVolt import MeasureVolt
import Driver.DataAcquisitionFpga.ContinousSequencerConfig as CsCfg

import logging
import time


class ContinousSequencer(Sequencer, MeasureVolt):
    def __init__(self):
        """
        Initiates a fpga object using the init in FPGAInterfaceHandling
        """
        self.config = CsCfg
        super(Sequencer, self).__init__(self.config.bitfilePath, self.config.bitfileSignature, self.config.fpgaResource)
        self.confHostBufferSize(self.config.transferToHostReqEle)
        self.type = self.config.seq_type

    '''read Indicators'''

    def getDACQuWriteTimeout(self):
        """
        function to check the DACQuWriteTimeout indicator which indicates if the DAC has timed out while
        writing to the Target-to-Host Fifo
        :return: bool, True if timedout
        """
        return self.ReadWrite(self.config.DACQuWriteTimeout).value

    def getSPCtrQuWriteTimeout(self):
        """
        :return: bool, timeout indicator of the Simple Counter trying to write to the DMAQueue
        """
        return self.ReadWrite(self.config.SPCtrQuWriteTimeout).value

    def getSPerrorCount(self):
        """
        :return: int, the errorCount of the simpleCounter module on the fpga
        """
        return self.ReadWrite(self.config.SPerrorCount).value

    def getSPState(self):
        """
        :return:int, state of SimpleCounter Module
        """
        return self.ReadWrite(self.config.SPstate).value

    '''set Controls'''

    def setDwellTime(self, scanParsDict, track_num):
        """
        set the dwell time for the continous sequencer.
        """
        track_name = 'track' + str(track_num)
        self.ReadWrite(self.config.dwellTime10ns, int(scanParsDict[track_name]['dwellTime10ns']))
        return self.checkFpgaStatus()

    def setAllContSeqPars(self, scanpars, track_num):
        """
        Set all Scanparameters, needed for the continousSequencer
        :param scanpars: dict, containing all scanparameters
        :return: bool, if success
        """
        track_name = 'track' + str(track_num)
        if self.changeSeqState(self.config.seqStateDict['idle']):
            if (self.setDwellTime(scanpars, track_num) and
                    self.setmeasVoltParameters(scanpars['measureVoltPars']) and
                    self.setTrackParameters(scanpars) and
                    self.set_trigger(scanpars[track_name].get('trigger', {})) and
                    self.selectKepcoOrScalerScan(scanpars['isotopeData']['type'])):
                return self.checkFpgaStatus()
        return False

    '''perform measurements:'''

    def measureOffset(self, scanpars, track_num):
        """
        set all scanparameters at the fpga and go into the measure Offset state.
        What the Fpga does then to measure the Offset is:
         set DAC to 0V
         set HeinzingerSwitchBox to the desired Heinzinger.
         send a pulse to the DMM
         wait until timeout/feedback from DMM
         done
         changed to state 'measComplete'
        Note: not included in Version 1 !
        :return:bool, True if successfully changed State
        """
        if self.setAllContSeqPars(scanpars, track_num):
            return self.changeSeqState(self.config.seqStateDict['measureOffset'])

    def measureTrack(self, scanpars, track_num):
        """
        set all scanparameters at the fpga and go into the measure Track state.
        Fpga will then measure one track independently from host and will finish either in
        'measComplete' or in 'error' state.
        In parallel, host has to read the data from the host sided buffer in parallel.
        :return:bool, True if successfully changed State
        """
        if self.setAllContSeqPars(scanpars, track_num):
            return self.changeSeqState(self.config.seqStateDict['measureTrack'])
        else:
            logging.debug('values could not be set')
