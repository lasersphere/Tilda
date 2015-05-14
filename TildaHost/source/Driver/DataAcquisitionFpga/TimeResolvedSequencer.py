"""

Created on '07.05.2015'

@author:'simkaufm'

"""

"""
Module in  charge for loading and accessing the TimeResolvedSequencer
Access Via the NiFpgaUniversalInterfaceDll.dll
"""

from Driver.DataAcquisitionFpga.FPGAInterfaceHandling import FPGAInterfaceHandling
import Driver.DataAcquisitionFpga.TimeResolvedSequencerConfig as TrsCfg
import time


class TimeResolvedSequencer(FPGAInterfaceHandling):
    def __init__(self):
        """
        initiates the FPGA, resetted and running.
        :return: None
        """
        self.TrsCfg = TrsCfg.TRSConfig()
        self.fpgaInterfaceInstance = super(TimeResolvedSequencer, self).__init__(self.TrsCfg.bitfilePath,
                                                                                 self.TrsCfg.bitfileSignature,
                                                                                 self.TrsCfg.fpgaResource)

    '''DMA Queue host sided Buffer Operations:'''
    def confHostBufferSize(self, nOfreqEle=-1):
        """
        Set the size of the host sided Buffer of "transferToHost" Fifo
         to the desired size as determined in the ...Config.py
        :param nOfreqEle: int, defines how many Elements should be capable in the Host sided Buffer.
        if nOfreqEle <= 0 the value from the Config.py will be taken.
        :return: True, if Status is no Error
        """
        if nOfreqEle <= 0:
            self.ConfigureU32FifoHostBuffer(self.TrsCfg.transferToHost['ref'], self.TrsCfg.transferToHost['nOfReqEle'])
            print('Size of Host Sided Buffer has been set to: ' + str(self.TrsCfg.transferToHost['nOfReqEle'])
                  + ' numbers of 32-Bit Elements')
        elif nOfreqEle > 0:
            self.ConfigureU32FifoHostBuffer(self.TrsCfg.transferToHost['ref'], nOfreqEle)
            print('Size of Host Sided Buffer has been set to: ' + str(nOfreqEle) + ' numbers of 32-Bit Elements')
        return self.checkFpgaStatus()

    def clearHostBuffer(self, nOfEle=-1):
        """
        Clear a certain number of Elements(nOfEle) from the Fifo.
        Should be called before closing the session!
        :param nOfEle: int, Number Of Elements that will be released from the Fifo.
        :return: bool, True if Status ok
        """
        return self.ClearU32FifoHostBuffer(self.TrsCfg.transferToHost['ref'], nOfEle)

    '''read Indicators:'''

    def getMCSState(self):
        """
        get the state of the MultiChannelScaler
        :return:int, state of MultiChannelScaler
        """
        return self.ReadWrite(self.TrsCfg.MCSstate).value

    def getSeqState(self):
        """
        get the state of the Sequencer
        :return:int, state of Sequencer
        """
        return self.ReadWrite(self.TrsCfg.seqState).value

    def getmeasVoltState(self):
        """
        gets the state of Voltage measurement Statemachine
        :return:int, state of Voltage measurement Statemachine
        """
        return self.ReadWrite(self.TrsCfg.measVoltState).value

    def getErrorCount(self):
        """
        gets the ErrorCount which represents how many errors have occured for the MultiChannelScaler
        for example each time abort is pressed, ErrorCount is raised by 1
        :return:int, MCSerrorcount
        """
        return self.ReadWrite(self.TrsCfg.MCSerrorcount).value

    def getDACQuWriteTimeout(self):
        """
        function to check the DACQuWriteTimeout indicator which indicates if the DAC has timed out while
        writing to the Target-to-Host Fifo
        :return: bool, True if timedout
        """
        return self.ReadWrite(self.TrsCfg.DACQuWriteTimeout).value

    '''set Controls'''

    def setCmdByHost(self, cmd):
        """
        send a command representing a desired state of the sequencer to the Ui of the Fpga
        :param cmd: int, desired sequencer State
        :return: int, cmd
        """
        return self.ReadWrite(self.TrsCfg.cmdByHost, cmd).value

    def changeSeqState(self, cmd, tries=-1, requestedState=-1):
        """
        Use this to change the state of the Sequencer. It will try to do so for number of tries.
        If State cannot be changed right now, it will return False.
        :param cmd: int, desired Sequencer State
        :param tries: int, number if tries so far, default is -1
        :param requestedState: int, variable for store if the command has already been sent, default is -1
        :return: bool, True for success, False if fail within number of maxTries.
        """
        maxTries = 10
        waitForNextTry = 0.001
        if tries > 0:
            time.sleep(waitForNextTry)
        curState = self.getSeqState()
        if curState == cmd:
            return self.checkFpgaStatus()
        elif tries == maxTries:
            print('could not Change to State ' + str(cmd) + ' within ' + str(maxTries)
                  + ' tries, Current State is: ' + str(curState))
            return False
        elif curState in [self.TrsCfg.seqState['measureOffset'], self.TrsCfg.seqState['measureTrack'],
                          self.TrsCfg.seqState['init']]:
            '''cannot change state while measuring or initializing, try again'''
            return self.changeSeqState(cmd, tries + 1, requestedState)
        elif curState in [self.TrsCfg.seqState['idle'], self.TrsCfg.seqState['measComplete'],
                          self.TrsCfg.seqState['error']]:
            '''send command to change state'''
            if requestedState < 0:
                '''only request to change state once'''
                self.setCmdByHost(cmd)
                return self.changeSeqState(cmd, tries + 1, cmd)
            else:
                return self.changeSeqState(cmd, tries + 1, requestedState)

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
        self.ReadWrite(self.TrsCfg.MCSSelectTrigger, mCSPars['MCSSelectTrigger'])
        self.ReadWrite(self.TrsCfg.delayticks, mCSPars['delayticks'])
        self.ReadWrite(self.TrsCfg.nOfBins, mCSPars['nOfBins'])
        self.ReadWrite(self.TrsCfg.nOfBunches, mCSPars['nOfBunches'])
        return self.checkFpgaStatus()

    def setmeasVoltParameters(self, measVoltPars):
        """
        Writes all values needed for the Voltage Measurement state machine to the fpga ui
        :param measVoltPars: dictionary, containing all necessary infos for Voltage measurement. These are:
        measVoltPulseLength25ns: long, Pulselength of the Trigger Pulse on PXI_Trig4 and CH
        measVoltTimeout10ns: long, timeout until which a response from the DMM must occur.
        :return: True if self.status == self.statusSuccess, else False
        """
        self.ReadWrite(self.TrsCfg.measVoltPulseLength25ns, measVoltPars['measVoltPulseLength25ns'])
        self.ReadWrite(self.TrsCfg.measVoltTimeout10ns, measVoltPars['measVoltTimeout10ns'])
        return self.checkFpgaStatus()

    def setTrackParameters(self, trackPars):
        """
        Writes all values needed for the Sequencer state machine to the fpga ui
        :param trackPars: dictionary, containing all necessary infos for measuring one track. These are:
        VoltOrScaler: bool, determine if the track is a KepcoScan or normal Scaler Scan
        stepSize: ulong, Stepsize for 18Bit-DAC Steps actually shifted by 2 so its 20 Bit Number
        start: ulong, Start Voltage for 18Bit-DAC actually shifted by 2 so its 20 Bit Number
        nOfSteps: long, Number Of Steps for one Track (=Scanregion)
        nOfScans: long, Number of Loops over this Track
        invertScan: bool, if True invert Scandirection on every 2nd Scan
        heinzingerControl: ubyte, Enum to determine which heinzinger will be active
        waitForKepco25nsTicks: uint, time interval after the voltage has been set and the unit waits
         before the scalers are activated. Unit is 25ns
        waitAfterReset25nsTicks: uint, time interval after the voltage has been reseted and the unit waits
         before the scalers are activated. Unit is 25ns
        :return: True if self.status == self.statusSuccess, else False
        """
        self.ReadWrite(self.TrsCfg.VoltOrScaler, trackPars['VoltOrScaler'])
        self.ReadWrite(self.TrsCfg.stepSize, trackPars['stepSize'])
        self.ReadWrite(self.TrsCfg.start, trackPars['start'])
        self.ReadWrite(self.TrsCfg.nOfSteps, trackPars['nOfSteps'])
        self.ReadWrite(self.TrsCfg.nOfScans, trackPars['nOfScans'])
        self.ReadWrite(self.TrsCfg.invertScan, trackPars['invertScan'])
        self.ReadWrite(self.TrsCfg.heinzingerControl, trackPars['heinzingerControl'])
        self.ReadWrite(self.TrsCfg.waitForKepco25nsTicks, trackPars['waitForKepco25nsTicks'])
        self.ReadWrite(self.TrsCfg.waitAfterReset25nsTicks, trackPars['waitAfterReset25nsTicks'])
        return self.checkFpgaStatus()

    def setAllScanParameters(self, scanpars):
        """
        Use the dictionary format of scanpars, to set all parameters at once.
         Therefore Sequencer must be in idle state
        :param scanpars: dictionary, containing all scanparameters
        :return: bool, True if successful
        """
        if self.changeSeqState(self.TrsCfg.seqState['idle']):
            if (self.setMCSParameters(scanpars) and self.setmeasVoltParameters(scanpars) and
                    self.setTrackParameters(scanpars)):
                return self.checkFpgaStatus()
        return False

    def abort(self):
        """
        abort the running execution immediatly
        :return: True if succes
        """
        i = 0
        imax = 500
        self.ReadWrite(self.TrsCfg.abort, True)
        while self.getSeqState() != self.TrsCfg.seqState['error'].value and i <= imax:
            time.sleep(0.001)
            i += 1
        self.ReadWrite(self.TrsCfg.abort, False)
        if self.getSeqState() == self.TrsCfg.seqState['error'].value:
            return self.getSeqState()
        else:
            return False

    def halt(self, val):
        """
        halts the Mesaruement after one loop is finished
        :return: True if success
        """
        self.ReadWrite(self.TrsCfg.halt, val)
        return self.checkFpgaStatus()

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
            return self.changeSeqState(self.TrsCfg.seqState['measureOffset'])

    def measureTrack(self, scanpars):
        """
        set all scanparameters at the fpga and go into the measure Track state.
        Fpga will then measure one track independently from host and will finish either in
        'measComplete' or in 'error' state.
        In parallel, host has to read the data from the host sided buffer in parallel.
        :return:bool, True if successfully changed State
        """
        if self.setAllScanParameters(scanpars):
            return self.changeSeqState(self.TrsCfg.seqState['measureTrack'])

    '''getting the data'''
    def getData(self):
        """
        read Data from host sided Buffer called 'transferToHost' to an Array.
        Can later be fed into a pipeline system.
        :return: python integer array with new data.
        """
        result = self.ReadU32Fifo(self.TrsCfg.transferToHost['ref'])
        return result

    '''closing and resetting'''
    def resetFpga(self):
        pass



# instanciate that bitch:
blub2 = TimeResolvedSequencer()

print('status of Fpga is: ' + str(blub2.status))
print('seq State: ' + str(blub2.getSeqState()))
print('configure Hist sided Buffer: ' + str(blub2.confHostBufferSize()))
time.sleep(0.1)

print('start Track: ' + str(blub2.measureTrack(blub2.TrsCfg.dummyScanParameters)))
print('seq State: ' + str(blub2.getSeqState()))
print('seq State: ' + str(blub2.getSeqState()))
print(blub2.getData())
print(blub2.getData())
print(blub2.getData())




# print(blub2.setCmdByHost(5))
# print(blub2.getSeqState())
# print(blub2.changeSeqState(3))
# print(blub2.changeSeqState(4))
# print(blub2.getSeqState())




# print(blub2.cmdByHost(5))
