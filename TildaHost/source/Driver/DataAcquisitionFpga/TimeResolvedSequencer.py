"""

Created on '07.05.2015'

@author:'simkaufm'


Module in  charge for loading and accessing the TimeResolvedSequencer
Access Via the NiFpgaUniversalInterfaceDll.dll
"""


from Driver.DataAcquisitionFpga.MeasureVolt import MeasureVolt
from Driver.DataAcquisitionFpga.SequencerCommon import Sequencer

import Driver.DataAcquisitionFpga.TimeResolvedSequencerConfig as TrsCfg


class TimeResolvedSequencer(Sequencer, MeasureVolt):
    def __init__(self):
        """
        initiates the FPGA, resetted and running.
        :return: None
        """

        self.config = TrsCfg
        super(Sequencer, self).__init__(self.config.bitfilePath, self.config.bitfileSignature, self.config.fpgaResource)
        self.confHostBufferSize(self.config)
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
    def setMCSParameters(self, mCSPars):
        """
        Writes all values needed for the Multi Channel Scaler state machine to the fpga ui
        :param mCSPars: dictionary, containing all necessary items for MCS. These are:
        MCSSelectTrigger: byte, Enum to select the active Trigger
        delayticks: ulong, Ticks to delay after triggered
        nOfBins: ulong, number of 10 ns bins that will be acquired per Trigger event
        nOfBunches: long, number of bunches that will be acquired per voltage Step
        :return: True if self.status == self.statusSuccess, else False
        """
        self.ReadWrite(self.config.MCSSelectTrigger, mCSPars['MCSSelectTrigger'])
        self.ReadWrite(self.config.delayticks, mCSPars['delayticks'])
        self.ReadWrite(self.config.nOfBins, mCSPars['nOfBins'])
        self.ReadWrite(self.config.nOfBunches, mCSPars['nOfBunches'])
        return self.checkFpgaStatus()

    def setAllScanParameters(self, scanpars):
        """
        Use the dictionary format of act_scan_wins, to set all parameters at once.
         Therefore Sequencer must be in idle state
        :param scanpars: dictionary, containing all scanparameters
        :return: bool, True if successful
        """
        if self.changeSeqState(self.config, self.config.seqStateDict['idle']):
            if (self.setMCSParameters(scanpars) and self.setmeasVoltParameters(self.config, scanpars) and
                    self.setTrackParameters(self.config, scanpars)):
                return self.checkFpgaStatus()
        return False

    '''perform measurements:'''

    def measureOffset(self, scanpars):
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
        if self.setAllScanParameters(scanpars):
            return self.changeSeqState(self.config, self.config.seqStateDict['measureOffset'])

    def measureTrack(self, scanpars):
        """
        set all scanparameters at the fpga and go into the measure Track state.
        Fpga will then measure one track independently from host and will finish either in
        'measComplete' or in 'error' state.
        In parallel, host has to read the data from the host sided buffer in parallel.
        :return:bool, True if successfully changed State
        """
        if self.setAllScanParameters(scanpars):
            return self.changeSeqState(self.config, self.config.seqStateDict['measureTrack'])


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
