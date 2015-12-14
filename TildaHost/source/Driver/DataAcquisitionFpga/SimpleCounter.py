"""
Created on 

@author: simkaufm

Module Description: Driver Interface for the Simple Counter bitfile,
 which should read all 8 PMTs with a dwelltime of 200 ms and
 should send all countervalues to the host via DMAQueue
"""

from Driver.DataAcquisitionFpga.FPGAInterfaceHandling import FPGAInterfaceHandling
import Driver.DataAcquisitionFpga.SimpleCounterConfig as ScCfg
import Service.VoltageConversions.VoltageConversions as VCon

import time


class SimpleCounter(FPGAInterfaceHandling):
    def __init__(self):
        self.type = 'sc'
        bit_path = ScCfg.bitfilePath
        bit_sig = ScCfg.bitfileSignature
        res = ScCfg.fpgaResource
        super(SimpleCounter, self).__init__(bit_path, bit_sig, res)
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
