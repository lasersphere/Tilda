"""
Created on 

@author: simkaufm

Module Description: Driver Interface for the Simple Counter bitfile,
 which should read all 8 PMTs with a dwelltime of 200 ms and
 should send all countervalues to the host via DMAQueue
"""
import logging

from Driver.DataAcquisitionFpga.FPGAInterfaceHandling import FPGAInterfaceHandling
import Driver.DataAcquisitionFpga.SimpleCounterConfig as ScCfg
import Service.VoltageConversions.VoltageConversions as VCon
from Driver.DataAcquisitionFpga.SequencerCommon import Sequencer
from Driver.DataAcquisitionFpga.TriggerTypes import TriggerTypes as TiTs


import time


class SimpleCounter(Sequencer):
    def __init__(self):
        self.type = 'sc'
        self.config = ScCfg
        bit_path = self.config.bitfilePath
        bit_sig = self.config.bitfileSignature
        res = self.config.fpgaResource
        super(Sequencer, self).__init__(bit_path, bit_sig, res)
        #super(SimpleCounter, self).__init__(bit_path, bit_sig, res)
        self.conf_host_buf(ScCfg.transferToHostReqEle)

    def conf_host_buf(self, num_of_request_ele):
        self.ConfigureU32FifoHostBuffer(ScCfg.transferToHost['ref'], num_of_request_ele)
        return self.checkFpgaStatus()

    def read_data_from_fifo(self):
        """
        :return: {nOfEle,  newData, elemRemainInFifo}
        nOfEle = int, number of Read Elements, newData = numpy.ndarray containing all data that was read
               elemRemainInFifo = int, number of Elements still in FifoBuffer
       """
        read_dict = self.ReadU32Fifo(ScCfg.transferToHost['ref'])
        return read_dict

    def set_dac_voltage(self, volt_dbl):
        dac_state = self.ReadWrite(ScCfg.DacState)
        t = 0
        t_max = 100
        while dac_state != ScCfg.dacState['idle'] and t < t_max:
            self.ReadWrite(ScCfg.DacStateCmdByHost, ScCfg.dacState['idle'])
            dac_state = self.ReadWrite(ScCfg.DacState)
            time.sleep(0.01)
            t += 1
        dac_reg_entry = VCon.get_24bit_input_from_voltage(volt_dbl)
        self.ReadWrite(ScCfg.setDACRegister, dac_reg_entry)
        self.ReadWrite(ScCfg.DacStateCmdByHost, ScCfg.dacState['setVolt'])
        return self.checkFpgaStatus()

    def set_post_acc_control(self, state_name):
        state_num = ScCfg.postAccOffsetVoltStateDict.get(state_name)
        if state_num is not None:
            self.ReadWrite(ScCfg.postAccOffsetVoltControl, state_num)

    def get_post_acc_control(self):
        """
        reads the state off the postacceleration control
        :return: post_acc_state, post_acc_name
        """
        post_acc_state = self.ReadWrite(ScCfg.postAccOffsetVoltState).value
        post_acc_name = ''
        for key, val in ScCfg.postAccOffsetVoltStateDict.items():
            if val == post_acc_state:
                post_acc_name = key
        return post_acc_state, post_acc_name

    def set_all_simpCnt_parameters(self, cntpars):
        """
        all parameters needed for the simple counting are set here
        """
        self.set_trigger(cntpars.get('timing', {}).get('trigger', {}))
        return self.checkFpgaStatus()

    ''' performe measurement '''
    def measure(self, cnt_pars):
        """
        set all counting parameters on the FPGA and go into counting state.
        FPGA will then measure counts independently from host until host says stop
        In parallel, host has to read the data from the host sided buffer in parallel.
        :param cnt_pars: dict: trigger parameters
        :return: bool: True if successfully changed State
        """
        if self.set_all_simpCnt_parameters(cnt_pars):
            #return self.changeSeqState(self.config.seqStateDict['measureTrack'])
            return True
        else:
            logging.DEBUG('trigger values for simple counter could not be set')

    #TODO: adjust this function to existing attributes in SimpleCounterConfig
    def set_trigger(self, trigger_dict=None):
        """
        sets all parameters related to the trigger.
        :param trigger_type: enum, defined in TriggerTypes.py
        :param trigger_dict: dict, containing all values needed for this type of trigger
        :return: True if success
        """
        meas_trigger_controls = {'triggerTypes': self.config.triggerTypes,
                              'selectTrigger': self.config.selectTrigger,
                              'trigDelay10ns': self.config.trigDelay10ns,
                              'triggerEdge': self.config.triggerEdge,
                              'softwareTrigger': self.config.softwareTrigger}

        trig_fpga_status = True
        for triggers, trig_dicts in trigger_dict.items():
            controls = {}
            if triggers == 'meas_trigger': controls = meas_trigger_controls
            #elif triggers == 'step_trigger': controls = step_trigger_controls
            #elif triggers == 'scan_trigger': controls = scan_trigger_controls

            trigger_type = trig_dicts.get('type', TiTs.no_trigger)
            logging.debug('setting trigger type to: ' + str(trigger_type) + ' value: ' + str(trigger_type.value))
            logging.debug('trigger dict is: ' + str(trig_dicts))
            self.ReadWrite(controls['triggerTypes'], trigger_type.value)
            if trigger_type is TiTs.no_trigger:
                trig_fpga_status = trig_fpga_status and self.checkFpgaStatus()
            elif trigger_type is TiTs.single_hit_delay:
                self.ReadWrite(controls['selectTrigger'], trig_dicts.get('trigInputChan', 0))
                self.ReadWrite(controls['trigDelay10ns'], int(trig_dicts.get('trigDelay10ns', 0)))
                trig_num = ['either', 'rising', 'falling'].index(trig_dicts.get('trigEdge', 'rising'))
                logging.debug('triggernum is: %s' % trig_num)
                self.ReadWrite(controls['triggerEdge'], trig_num)
                trig_fpga_status = trig_fpga_status and self.checkFpgaStatus()
        return trig_fpga_status