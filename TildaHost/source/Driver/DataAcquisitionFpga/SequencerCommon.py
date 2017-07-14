"""

Created on '08.07.2015'

@author:'simkaufm'

"""
import logging
import time

import Service.VoltageConversions.VoltageConversions as VCon
from Driver.DataAcquisitionFpga.FPGAInterfaceHandling import FPGAInterfaceHandling
from Driver.DataAcquisitionFpga.TriggerTypes import TriggerTypes as TiTs


class Sequencer(FPGAInterfaceHandling):
    '''DMA Queue host sided Buffer Operations:'''

    def confHostBufferSize(self, nOfreqEle=-1):
        """
        Set the size of the host sided Buffer of "transferToHost" Fifo
         to the desired size as determined in the ...self.config.py
        :param nOfreqEle: int, defines how many Elements should be capable in the Host sided Buffer.
        if nOfreqEle <= 0 the value from the self.config.py will be taken.
        default is -1
        :return: True, if Status is no Error
        """
        if nOfreqEle <= 0:
            self.ConfigureU32FifoHostBuffer(self.config.transferToHost['ref'], self.config.transferToHostReqEle)
            print('Size of Host Sided Buffer has been set to: ' + str(self.config.transferToHostReqEle)
                  + ' numbers of 32-Bit Elements')
        elif nOfreqEle > 0:
            self.ConfigureU32FifoHostBuffer(self.config.transferToHost['ref'], nOfreqEle)
            print('Size of Host Sided Buffer has been set to: ' + str(nOfreqEle) + ' numbers of 32-Bit Elements')
        return self.checkFpgaStatus()

    def clearHostBuffer(self, nOfEle=-1):
        """
        Clear a certain number of Elements(nOfEle) from the Fifo.
        Should be called before closing the session!
        :param nOfEle: int, Number Of Elements that will be released from the Fifo.
        :return: bool, True if Status ok
        """
        return self.ClearU32FifoHostBuffer(self.config.transferToHost['ref'], nOfEle)

    '''reading'''

    def getSeqState(self):
        """
        get the state of the Sequencer
        :return:int, state of Sequencer
        """
        return self.ReadWrite(self.config.seqState).value

    def getHeinzControlState(self):
        """read the state of the post Acceleration Control Box"""
        ret = self.ReadWrite(self.config.postAccOffsetVoltState).value
        # logging.debug('HSB-State is: ' + str(ret))
        return ret

    def getPostAccelerationControlStateIsDone(self, desired_state):
        """
        call this to check if the state of the hsb is already the desired one.
        :param desired_state: int, the desired state of the box
        :return: tuple, (bool_True_if_success, int_current_state, int_desired_state)
        """
        currentState = int(self.getHeinzControlState())
        done = currentState == desired_state
        # logging.debug('switchbox, state: %s, desired state: %s, done: %s' % (currentState, desired_state, done))
        return done, currentState, desired_state

    '''writing'''

    def setTrackParameters(self, trackPars):
        """
        Writes all values needed for the Sequencer state machine to the fpga ui
        :param trackPars: dictionary, containing all necessary infos for measuring one track. These are:
        VoltOrScaler: bool, determine if the track is a KepcoScan or normal Scaler Scan
        dacStepSize18Bit: ulong, Stepsize for 18Bit-DAC Steps actually shifted by 2 so its 20 Bit Number
        dacStartRegister18Bit: ulong, Start Voltage for 18Bit-DAC actually shifted by 2 so its 20 Bit Number
        nOfSteps: long, Number Of Steps for one Track (=Scanregion)
        nOfScans: long, Number of Loops over this Track
        invertScan: bool, if True invert Scandirection on every 2nd Scan
        postAccOffsetVoltControl: ubyte, Enum to determine which heinzinger will be active
        waitForKepco1us: u32b, time interval after the voltage has been set and the unit waits
         before the scalers are activated. Unit is 1us
        waitAfterReset1us: u32b, time interval after the voltage has been reseted and the unit waits
         before the scalers are activated. Unit is 1us
        :return: True if self.status == self.statusSuccess, else False
        """

        self.ReadWrite(self.config.dacStepSize18Bit, trackPars['dacStepSize18Bit'])
        self.ReadWrite(self.config.dacStartRegister18Bit, trackPars['dacStartRegister18Bit'])
        self.ReadWrite(self.config.nOfSteps, trackPars['nOfSteps'])
        self.ReadWrite(self.config.nOfScans, trackPars['nOfScans'])
        self.ReadWrite(self.config.invertScan, trackPars['invertScan'])
        self.ReadWrite(self.config.waitForKepcous, trackPars.get('waitForKepco1us', 50))
        self.ReadWrite(self.config.waitAfterResetus, trackPars.get('waitAfterReset1us', 500))
        # self.setPostAccelerationControlState(trackPars['postAccOffsetVoltControl'], True)
        return self.checkFpgaStatus()

    def selectKepcoOrScalerScan(self, typestr):
        logging.debug('type of scan: ' + typestr)
        if typestr == 'kepco':
            print('this is a kepco scan')
            self.ReadWrite(self.config.VoltOrScaler, True)
        else:
            self.ReadWrite(self.config.VoltOrScaler, False)
        return self.checkFpgaStatus()

    def setPostAccelerationControlState(self, desiredState, blocking=True):
        """
        will set the PostAccelerationControl State, so one can chose which PowerSupply will be used.
        :return: int, the current State of the Control Box
        """
        currentState = self.getHeinzControlState()
        if currentState != desiredState:
            if self.changeSeqState(self.config.seqStateDict['idle']):
                self.ReadWrite(self.config.postAccOffsetVoltControl, desiredState)
                timeout = 0
                while blocking and timeout < 30:
                    done, currentState, desired_state = self.getPostAccelerationControlStateIsDone(desiredState)
                    print('currentState:', currentState, '\tdesiredState: ', desiredState)
                    if done:
                        return currentState
                    else:
                        time.sleep(0.2)
                        timeout += 1
            else:
                raise Exception('could not change the fpga state to idle for setting the switch box')
        else:
            return currentState

    def setCmdByHost(self, cmd):
        """
        send a command representing a desired state of the sequencer to the Ui of the Fpga
        :param cmd: int, desired sequencer State
        :return: int, cmd
        """
        return self.ReadWrite(self.config.cmdByHost, cmd).value

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
        waitForNextTry = 0.01
        if tries > 0:
            time.sleep(waitForNextTry)
        curState = self.getSeqState()
        if curState == cmd:
            state_name = [state_n for state_n, state_num in self.config.seqStateDict.items() if state_num == curState][
                0]
            logging.debug('fpga states successfully changed to: %s <-> %s' % (curState, state_name))
            return self.checkFpgaStatus()
        elif tries == maxTries:
            print('could not Change to State ' + str(cmd) + ' within ' + str(maxTries) +
                  ' tries, Current State is: ' + str(curState))
            return False
        elif curState in [self.config.seqStateDict['measureTrack'],
                          self.config.seqStateDict['init']]:
            '''cannot change state while measuring or initializing, try again'''
            return self.changeSeqState(cmd, tries + 1, requestedState)
        elif curState in [self.config.seqStateDict['idle'], self.config.seqStateDict['measComplete'],
                          self.config.seqStateDict['error'], self.config.seqStateDict['measureOffset']]:
            '''send command to change state'''
            try:
                if requestedState < 0:
                    '''only request to change state once'''
                    self.setCmdByHost(cmd)
                    return self.changeSeqState(cmd, tries + 1, cmd)
                else:
                    return self.changeSeqState(cmd, tries + 1, requestedState)
            except Exception as e:
                print('error while changing state:', e, 'cmd: ', cmd, 'curState: ', curState)

    def abort(self):
        """
        abort the running execution immediately, will block until state is 'error' or timedout.
        :return: True if success
        """
        i = 0
        imax = 500
        self.ReadWrite(self.config.abort, True)
        while self.getSeqState() != self.config.seqStateDict['error'] and i <= imax:
            time.sleep(0.001)
            i += 1
        self.ReadWrite(self.config.abort, False)
        if self.getSeqState() == self.config.seqStateDict['error']:
            return self.getSeqState()
        else:
            return False

    def halt(self, val):
        """
        halts the Mesaruement after one loop is finished
        :return: True if success
        """
        print('setting halt to: ', val)
        self.ReadWrite(self.config.halt, val)
        return self.checkFpgaStatus()

    def pause_scan(self, pause_bool=None):
        """
        This will pause the scan with a loop in the handshake.
        Use this, if the laser jumped or so and you want to continue on the data.
        :param pause_bool: bool, None if you want to toggle
        """
        if pause_bool is None:
            pause_bool = not self.pause_bool
        print('pausing the fpga, pause is: %s' % pause_bool)
        self.ReadWrite(self.config.pause, pause_bool)
        stat = self.checkFpgaStatus()
        if stat:
            self.pause_bool = pause_bool
        return stat

    def set_trigger(self, trigger_dict=None):
        """
        sets all parameters related to the trigger.
        :param trigger_type: enum, defined in TriggerTypes.py
        :param trigger_dict: dict, containing all values needed for this type of trigger
        :return: True if success
        """
        trigger_type = trigger_dict.get('type', TiTs.no_trigger)
        logging.debug('setting trigger type to: ' + str(trigger_type) + ' value: ' + str(trigger_type.value))
        logging.debug('trigger dict is: ' + str(trigger_dict))
        self.ReadWrite(self.config.triggerTypes, trigger_type.value)
        if trigger_type is TiTs.no_trigger:
            return self.checkFpgaStatus()
        elif trigger_type is TiTs.single_hit_delay:
            self.ReadWrite(self.config.selectTrigger, trigger_dict.get('trigInputChan', 0))
            self.ReadWrite(self.config.trigDelay10ns, int(trigger_dict.get('trigDelay10ns', 0)))
            trig_num = ['either', 'rising', 'falling'].index(trigger_dict.get('trigEdge', 'rising'))
            print('triggernum is: ', trig_num)
            self.ReadWrite(self.config.triggerEdge, trig_num)
            return self.checkFpgaStatus()
        elif trigger_type is TiTs.single_hit:
            trig_num = ['either', 'rising', 'falling'].index(trigger_dict.get('trigEdge', 'rising'))
            print('triggernum is: ', trig_num)
            self.ReadWrite(self.config.triggerEdge, trig_num)
            self.ReadWrite(self.config.selectTrigger, trigger_dict.get('trigInputChan', 0))
            return self.checkFpgaStatus()

    def set_0volt_dac_register(self, null_volt=None):
        """
        function to set the 0V DAC register to 0 Volts as gained by the calibration
        :return: True if success
        """
        if null_volt is None:
            null_volt = VCon.get_24bit_input_from_voltage(0)
        self.ReadWrite(self.config.dac0VRegister, null_volt)
        return self.checkFpgaStatus()

    '''getting the data'''

    def getData(self):
        """
        read Data from host sided Buffer called 'transferToHost' to an Array.
        Can later be fed into a pipeline system.
        :return: dictionary,
        nOfEle = int, number of Read Elements, newData = numpy Array containing all data that was read
               elemRemainInFifo = int, number of Elements still in FifoBuffer
        """
        result = self.ReadU32Fifo(self.config.transferToHost['ref'])
        result.update(newData=result['newData'])
        return result
