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
        self.fullRawData = []
        self.TrsCfg = TrsCfg.TRSConfig()
        self.fpgaInterfaceInstance = super(TimeResolvedSequencer, self).__init__(self.TrsCfg.bitfilePath, self.TrsCfg.bitfileSignature, self.TrsCfg.fpgaResource)

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
            return True
        elif tries == maxTries:
            print('could not Change to State ' +str(cmd) + ' within ' + str(maxTries)
             + ' tries, Current State is: ' + str(curState))
            return False         
        elif curState in [self.TrsCfg.seqState['measureOffset'], self.TrsCfg.seqState['measureTrack'],
                           self.TrsCfg.seqState['init']]:
            '''cannot change state while measuring or initializing, try again'''
            return self.changeSeqState(cmd, tries+1, requestedState)
        elif curState in [self.TrsCfg.seqState['idle'], self.TrsCfg.seqState['measComplete'],
                          self.TrsCfg.seqState['error']]:
            '''send command to change state'''
            if requestedState < 0:
                '''only request to change state once'''
                self.setCmdByHost(cmd)
                return self.changeSeqState(cmd, tries+1, cmd)
            else:
                return self.changeSeqState(cmd, tries+1, requestedState)

    def setMCSValues(self, mCSPars):
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

    def setmeasVoltValues(self, measVoltPars):
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

    def setTrackValues(self, trackPars):
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
    def measureOffset(self):
        """
        If this is called, the Host should already have set the chosen Heinzinger to the desired Voltage via SCPI.
        What the Fpga doeas than to measure the Offset is:
         set DAC to 0V
         set HeinzingerSwitchBox to the desired Heinzinger.
         send a pulse to the DMM
         wait until timeout/feedback from DMM
         done
        Note: not included in V1 !
        :return: True if success
        """
        return True

    def measureTrack(self):
        """
        measure one Track with all Parameters already passed to the fpga before.
        :return: True if success
        """
        timeBetweenStateChecks = 0.01
        if self.getSeqState() in [self.TrsCfg.seqState['idle'], self.TrsCfg.seqState['measComplete'],
                                  self.TrsCfg.seqState['error']]:
            self.setCmdByHost(self.TrsCfg.seqState['measureTrack'])
            if self.checkFpgaStatus():
                while self.getSeqState() == self.TrsCfg.seqState['measureTrack']:
                    self.getData()
                    time.sleep(timeBetweenStateChecks)
            if self.getSeqState() == self.TrsCfg.seqState['measComplete']:
                return self.checkFpgaStatus()
            else:
                return False
        else:
            return False

    def getData(self):
        """
        read Data from host sided Buffer to an Array
        :return:
        """
        result = self.ReadU32Fifo(self.TrsCfg.transferToHost['ref'])

        self.fullRawData.append(result['newData'])








    def startTrack(self, scanPars):
        """
        tbw
        :param scanPars: dictionary, containing all parameters needed for a scan.
        :return:
        """
        if self.changeSeqState(1):
            '''must change state to idle in order to transfer ui-values to global vars on the fpga'''
            self.setMCSValues(scanPars)
            self.setmeasVoltValues(scanPars)
            self.setTrackValues(scanPars)
            if scanPars['measureOffset']:
                '''change state to measure the Offset before scanning the track'''
                while not self.measureOffset(scanPars):
                    pass
            self.measureTrack()


        else:
            print('could not change into idle state, Maybe abort running tasks? Otherwise reset Bitfile.')
            return False



# instanciate that bitch:
blub2 = TimeResolvedSequencer()
print('init: ' + str(blub2.__init__()))
print('status of Fpga is: ' + str(blub2.status))
print(blub2.getSeqState())
time.sleep(0.1)
print(blub2.setCmdByHost(5))
print(blub2.getSeqState())
# print(blub2.changeSeqState(3))
# print(blub2.changeSeqState(4))
# print(blub2.getSeqState())




# print(blub2.cmdByHost(5))
