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
        self.fpgaInterfaceInstance = super(TimeResolvedSequencer, self).__init__(self.TrsCfg.bitfilePath, self.TrsCfg.bitfileSignature, self.TrsCfg.fpgaResource)

    def getMCSState(self):
        """
        get the state of the MultiChannelScaler
        :return:int, state of MultiChannelScaler
        """
        self.ReadWrite(self.TrsCfg.MCSstate)
        return self.TrsCfg.MCSstate['val'].value

    def getSeqState(self):
        """
        get the state of the Sequencer
        :return:int, state of Sequencer
        """
        self.ReadWrite(self.TrsCfg.seqState)
        return self.TrsCfg.seqState['val'].value

    def getmeasVoltState(self):
        """
        gets the state of Voltage measurement Statemachine
        :return:int, state of Voltage measurement Statemachine
        """
        self.ReadWrite(self.TrsCfg.measVoltState)
        return self.TrsCfg.measVoltState['val'].value

    def getErrorCount(self):
        """
        gets the ErrorCount which represents how many errors have occured for the MultiChannelScaler
        for example each time abort is pressed, ErrorCount is raised by 1
        :return:int, MCSerrorcount
        """
        self.ReadWrite(self.TrsCfg.MCSerrorcount)
        return self.TrsCfg.MCSerrorcount['val'].value

    def getDACQuWriteTimeout(self):
        """
        function to check the DACQuWriteTimeout indicator which indicates if the DAC has timed out while
        writing to the Target-to-Host Fifo
        :return: bool, True if timedout
        """
        self.ReadWrite(self.TrsCfg.DACQuWriteTimeout)
        return self.TrsCfg.DACQuWriteTimeout['val'].value

    def setCmdByHost(self, cmd):
        """
        send a command representing a desired state of the sequencer to the Ui of the Fpga
        :param cmd: int, desired sequencer State
        :return: bool, True if returned cmd equals the requested one and there is  no Error and
        """
        self.TrsCfg.cmdByHost['val'].value = cmd
        self.ReadWrite(self.TrsCfg.cmdByHost)
        if self.TrsCfg.cmdByHost['val'].value == cmd and self.status == self.statusSuccess:
            return True
        else:
            print('could not send command to Fpga: '+ str(cmd) 
                  + 'status is: ' + str(self.status))
            return False

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

    def setMCSValues(self, selectTrigger, delay, nOfBins, nOfBunches):
        """
        tbw
        :param selectTrigger:
        :param delay:
        :param nOfBins:
        :param nOfBunches:
        :return: bool, True is successful
        """
        pass

    def setmeasVoltValues(self, pulselength, timeout):
        """
        tbw
        :param pulselength:
        :param timeout:
        :return: bool, True is successful
        """
        pass

    def setTrackValues(self, voltOrScaler, stepSize, startVoltage, nOfSteps, nOfScans,
                       invertScan, whichHeinzinger, waitForKepco, waitAfterReset):
        """
        tbw
        :param voltOrScaler:
        :param stepSize:
        :param startVoltage:
        :param nOfSteps:
        :param nOfScans:
        :param invertScan:
        :param whichHeinzinger:
        :param waitForKepco:
        :param waitAfterReset:
        :return: bool, True is successful
        """
        pass

    def startScan(self, scanPars):
        """
        tbw
        :param scanPars: dictionary, containing all parameters needed for a scan.
        :return:
        """
        if self.changeSeqState(1):
            '''must change state to idle in order to transfer ui-values to global vars on the fpga'''
            self.setMCSValues(scanPars['selectTrigger'], scanPars['delay'], scanPars['nOfBins'], scanPars['nofBunches'])
            self.setmeasVoltValues(scanPars['measVoltPulseLength25ns'], scanPars['measVoltTimeout10ns'])
            self.setTrackValues('keep working here')
        else:
            print('could not change into idle state, Maybe abort running tasks? Otherwise reset Bitfile.')
            return False



# instanciate that bitch:
blub2 = TimeResolvedSequencer()
print('init: ' + str(blub2.__init__()))
print(blub2.getSeqState())
time.sleep(0.1)
print(blub2.getSeqState())
print(blub2.changeSeqState(3))
print(blub2.changeSeqState(4))
print(blub2.getSeqState())




# print(blub2.cmdByHost(5))
