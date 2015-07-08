"""

Created on '08.07.2015'

@author:'simkaufm'

"""
from Driver.DataAcquisitionFpga.FPGAInterfaceHandling import FPGAInterfaceHandling

import time

class Sequencer(FPGAInterfaceHandling):

    '''DMA Queue host sided Buffer Operations:'''
    def confHostBufferSize(self, config, nOfreqEle=-1):
        """
        Set the size of the host sided Buffer of "transferToHost" Fifo
         to the desired size as determined in the ...Config.py
        :param nOfreqEle: int, defines how many Elements should be capable in the Host sided Buffer.
        if nOfreqEle <= 0 the value from the Config.py will be taken.
        default is -1
        :return: True, if Status is no Error
        """
        if nOfreqEle <= 0:
            self.ConfigureU32FifoHostBuffer(config.transferToHost['ref'], config.transferToHostReqEle)
            print('Size of Host Sided Buffer has been set to: ' + str(config.transferToHostReqEle)
                  + ' numbers of 32-Bit Elements')
        elif nOfreqEle > 0:
            self.ConfigureU32FifoHostBuffer(config.transferToHost['ref'], nOfreqEle)
            print('Size of Host Sided Buffer has been set to: ' + str(nOfreqEle) + ' numbers of 32-Bit Elements')
        return self.checkFpgaStatus()

    def clearHostBuffer(self, config, nOfEle=-1):
        """
        Clear a certain number of Elements(nOfEle) from the Fifo.
        Should be called before closing the session!
        :param nOfEle: int, Number Of Elements that will be released from the Fifo.
        :return: bool, True if Status ok
        """
        return self.ClearU32FifoHostBuffer(config.transferToHost['ref'], nOfEle)


    '''reading'''
    def getSeqState(self, config):
        """
        get the state of the Sequencer
        :return:int, state of Sequencer
        """
        return self.ReadWrite(config.seqState).value


    '''writing'''
    def setTrackParameters(self, config, trackPars):
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
        self.ReadWrite(config.VoltOrScaler, trackPars['VoltOrScaler'])
        self.ReadWrite(config.stepSize, trackPars['stepSize'])
        self.ReadWrite(config.start, trackPars['start'])
        self.ReadWrite(config.nOfSteps, trackPars['nOfSteps'])
        self.ReadWrite(config.nOfScans, trackPars['nOfScans'])
        self.ReadWrite(config.invertScan, trackPars['invertScan'])
        self.ReadWrite(config.heinzingerControl, trackPars['heinzingerControl'])
        self.ReadWrite(config.waitForKepco25nsTicks, trackPars['waitForKepco25nsTicks'])
        self.ReadWrite(config.waitAfterReset25nsTicks, trackPars['waitAfterReset25nsTicks'])
        return self.checkFpgaStatus()

    def setCmdByHost(self, config, cmd):
        """
        send a command representing a desired state of the sequencer to the Ui of the Fpga
        :param cmd: int, desired sequencer State
        :return: int, cmd
        """
        return self.ReadWrite(config.cmdByHost, cmd).value

    def changeSeqState(self, config, cmd, tries=-1, requestedState=-1):
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
        curState = self.getSeqState(config)
        if curState == cmd:
            return self.checkFpgaStatus()
        elif tries == maxTries:
            print('could not Change to State ' + str(cmd) + ' within ' + str(maxTries)
                  + ' tries, Current State is: ' + str(curState))
            return False
        elif curState in [config.seqStateDict['measureOffset'], config.seqStateDict['measureTrack'],
                          config.seqStateDict['init']]:
            '''cannot change state while measuring or initializing, try again'''
            return self.changeSeqState(cmd, tries + 1, requestedState)
        elif curState in [config.seqStateDict['idle'], config.seqStateDict['measComplete'],
                          config.seqStateDict['error']]:
            '''send command to change state'''
            if requestedState < 0:
                '''only request to change state once'''
                self.setCmdByHost(config, cmd)
                return self.changeSeqState(config, cmd, tries + 1, cmd)
            else:
                return self.changeSeqState(config, cmd, tries + 1, requestedState)

    def abort(self, config):
        """
        abort the running execution immediatly
        :return: True if succes
        """
        i = 0
        imax = 500
        self.ReadWrite(config.abort, True)
        while self.getSeqState() != config.seqStateDict['error'].value and i <= imax:
            time.sleep(0.001)
            i += 1
        self.ReadWrite(config.abort, False)
        if self.getSeqState() == config.seqStateDict['error'].value:
            return self.getSeqState()
        else:
            return False

    def halt(self, config, val):
        """
        halts the Mesaruement after one loop is finished
        :return: True if success
        """
        self.ReadWrite(config.halt, val)
        return self.checkFpgaStatus()
