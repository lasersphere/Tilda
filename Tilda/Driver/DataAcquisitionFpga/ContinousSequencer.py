"""

Created on '09.07.2015'

@author:'simkaufm'

"""

import logging

import Tilda.Driver.DataAcquisitionFpga.ContinousSequencerConfig as CsCfg
from Tilda.Driver.DataAcquisitionFpga.MeasureVolt import MeasureVolt
from Tilda.Driver.DataAcquisitionFpga.OutBits import Outbits
from Tilda.Driver.DataAcquisitionFpga.SequencerCommon import Sequencer


class ContinousSequencer(Sequencer, MeasureVolt, Outbits):
    def __init__(self):
        """
        Initiates a fpga object using the init in FPGAInterfaceHandling
        """
        self.config = CsCfg
        super(Sequencer, self).__init__(self.config.bitfilePath, self.config.bitfileSignature, self.config.fpgaResource)
        self.confHostBufferSize(self.config.transferToHostReqEle)
        self.set_0volt_dac_register()
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

    def get_state(self):
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
        if scanParsDict['isotopeData']['type'] == 'kepco':
            dwell = 1000  # force this to 10µs if kepco scanning
        else:
            dwell = int(scanParsDict[track_name]['dwellTime10ns'])
        self.ReadWrite(self.config.dwellTime10ns, dwell)
        return self.checkFpgaStatus()

    def setAllContSeqPars(self, scanpars, track_num, pre_post_scan_meas_str):
        """
        Set all Scanparameters, needed for the continousSequencer
        :param scanpars: dict, containing all scanparameters
        :return: bool, if success
        """
        track_name = 'track' + str(track_num)
        if self.changeSeqState(self.config.seqStateDict['idle']):
            if (self.setDwellTime(scanpars, track_num) and
                    self.setmeasVoltParameters(scanpars[track_name]['measureVoltPars'][pre_post_scan_meas_str]) and
                    self.setTrackParameters(scanpars[track_name]) and
                    self.setScanDeviceParameters(scanpars[track_name].get('scanDevice', {})) and
                    self.set_trigger(scanpars[track_name].get('trigger', {})) and
                    self.selectKepcoOrScalerScan(scanpars['isotopeData']['type']) and
                    self.set_outbits_cmd(scanpars[track_name]['outbits'], pre_post_scan_meas_str)):
                return self.checkFpgaStatus()
        return False

    '''perform measurements:'''

    def measureOffset(self, scanpars, track_num, pre_post_scan_meas_str):
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
        if self.setAllContSeqPars(scanpars, track_num, pre_post_scan_meas_str):
            return self.changeSeqState(self.config.seqStateDict['measureOffset'])

    def measureTrack(self, scanpars, track_num):
        """
        set all scanparameters at the fpga and go into the measure Track state.
        Fpga will then measure one track independently from host and will finish either in
        'measComplete' or in 'error' state.
        In parallel, host has to read the data from the host sided buffer in parallel.
        :return:bool, True if successfully changed State
        """
        if self.setAllContSeqPars(scanpars, track_num, 'duringScan'):
            return self.changeSeqState(self.config.seqStateDict['measureTrack'])
        else:
            logging.debug('values could not be set')
