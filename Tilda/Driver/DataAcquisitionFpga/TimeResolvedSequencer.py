"""

Created on '07.05.2015'

@author:'simkaufm'


Module in  charge for loading and accessing the TimeResolvedSequencer
Access Via the NiFpgaUniversalInterfaceDll.dll
"""
import logging

import Tilda.Driver.DataAcquisitionFpga.TimeResolvedSequencerConfig as TrsCfg
from Tilda.Driver.DataAcquisitionFpga.MeasureVolt import MeasureVolt
from Tilda.Driver.DataAcquisitionFpga.SequencerCommon import Sequencer
from Tilda.Driver.DataAcquisitionFpga.OutBits import Outbits


class TimeResolvedSequencer(Sequencer, MeasureVolt, Outbits):
    def __init__(self):
        """
        initiates the FPGA, resetted and running.
        :return: None
        """

        self.config = TrsCfg
        super(Sequencer, self).__init__(self.config.bitfilePath, self.config.bitfileSignature, self.config.fpgaResource)
        self.confHostBufferSize(self.config.transferToHostReqEle)
        self.set_0volt_dac_register()
        self.type = self.config.seq_type

    '''read Indicators:'''

    def getMCSState(self):
        """
        get the state of the MultiChannelScaler
        :return:int, state of MultiChannelScaler
        """
        return self.ReadWrite(self.config.MCSstate).value

    def getDACQuWriteTimeout(self):
        """
        function to check the DACQuWriteTimeout indicator which indicates if the DAC has timed out while
        writing to the Target-to-Host Fifo
        :return: bool, True if timedout
        """
        return self.ReadWrite(self.config.DACQuWriteTimeout).value

    def getErrorCount(self):
        """
        gets the ErrorCount which represents how many errors have occured for the MultiChannelScaler
        for example each time abort is pressed, ErrorCount is raised by 1
        :return:int, MCSerrorcount
        """
        return self.ReadWrite(self.config.MCSerrorcount).value

    '''set Controls'''

    def setMCSParameters(self, scanpars, track_name):
        """
        Writes all values needed for the Multi Channel Scaler state machine to the fpga ui
        :param scanpars: dictionary, containing all necessary items for MCS. These are:
        nOfBins: ulong, number of 10 ns bins that will be acquired per Trigger event
        nOfBunches: long, number of bunches that will be acquired per voltage Step
        :return: True if self.status == self.statusSuccess, else False
        """
        if scanpars['isotopeData']['type'] == 'kepco':
            nofbins = 1000  # force these values when performing a kepco scan within the trs
            nofbunches = 1
        else:
            nofbins = scanpars[track_name]['nOfBins']
            nofbunches = scanpars[track_name]['nOfBunches']
        self.ReadWrite(self.config.nOfBins, nofbins)
        self.ReadWrite(self.config.nOfBunches, nofbunches)
        return self.checkFpgaStatus()

    def setAllScanParameters(self, scanpars, track_num, pre_post_scan_meas_str):
        """
        Use the dictionary format of act_scan_wins, to set all parameters at once.
         Therefore Sequencer must be in idle state
        :param scanpars: dictionary, containing all scanparameters
        :return: bool, True if successful
        """
        track_name = 'track' + str(track_num)
        if self.changeSeqState(self.config.seqStateDict['idle']):
            if (self.setMCSParameters(scanpars, track_name) and
                    self.setmeasVoltParameters(scanpars[track_name]['measureVoltPars'][pre_post_scan_meas_str]) and
                    self.setTrackParameters(scanpars[track_name]) and
                    self.set_trigger(scanpars[track_name].get('trigger', {})) and
                    self.setScanDeviceParameters(scanpars[track_name]['scanDevice']) and
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
        if self.setAllScanParameters(scanpars, track_num, pre_post_scan_meas_str):
            return self.changeSeqState(self.config.seqStateDict['measureOffset'])

    def measureTrack(self, scanpars, track_num):
        """
        set all scanparameters at the fpga and go into the measure Track state.
        Fpga will then measure one track independently from host and will finish either in
        'measComplete' or in 'error' state.
        In parallel, host has to read the data from the host sided buffer in parallel.
        :return:bool, True if successfully changed State
        """
        if self.setAllScanParameters(scanpars, track_num, 'duringScan'):
            return self.changeSeqState(self.config.seqStateDict['measureTrack'])
        else:
            logging.debug('values could not be set')

#
# blub2 = TimeResolvedSequencer()
#
# print('status of Fpga is: ' + str(blub2.status))
# print('seq State: ' + str(blub2.getSeqState()))
# print('configure Hist sided Buffer: ' + str(blub2.confHostBufferSize()))
# time.sleep(0.1)
#
# print('dacStartRegister18Bit Track: ' + str(blub2.measureTrack(blub2.self.config.dummyScanParameters)))
# print('seq State: ' + str(blub2.getSeqState()))
# print('seq State: ' + str(blub2.getSeqState()))
# print(blub2.getData())
# print(blub2.getData())
# print(blub2.getData())




# print(blub2.setCmdByHost(5))
# print(blub2.getSeqState())
# print(blub2.changeSeqState(3))
# print(blub2.changeSeqState(4))
# print(blub2.getSeqState())




# print(blub2.cmdByHost(5))
