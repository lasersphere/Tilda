"""
Created on 12.04.16

@author: simkaufm

Module Description:

Module for controling the bitfile of TildaPassive.
This bitfile is foreseen for DAQ in parallel to the Master Control Program (MCP)

"""

from Driver.DataAcquisitionFpga.FPGAInterfaceHandling import FPGAInterfaceHandling
import Driver.DataAcquisitionFpga.TildaPassiveConfig as TpCfg


class TildaPassive(FPGAInterfaceHandling):
    def __init__(self):
        self.type = 'tipa'
        bit_path = TpCfg.bitfilePath
        bit_sig = TpCfg.bitfileSignature
        res = TpCfg.fpgaResource
        super(TildaPassive, self).__init__(bit_path, bit_sig, res)
        self.conf_host_buf(TpCfg.transferToHostReqEle)
        self.set_bin_num(TpCfg.default_nOfBins)
        self.set_delay(TpCfg.default_delay)

    def conf_host_buf(self, num_of_request_ele):
        """
        configure the host buffer which basically only means
        setting the number of elements that the host will be able store at once.
        """
        self.ConfigureU32FifoHostBuffer(TpCfg.transferToHost['ref'], num_of_request_ele)
        return self.checkFpgaStatus()

    def read_data_from_fifo(self):
        """
        :return: {nOfEle,  newData, elemRemainInFifo}
        nOfEle = int, number of Read Elements, newData = numpy.ndarray containing all data that was read
               elemRemainInFifo = int, number of Elements still in FifoBuffer
       """
        read_dict = self.ReadU32Fifo(TpCfg.transferToHost['ref'])
        return read_dict

    def read_tilda_passive_status(self):
        """
        read the status of the statemachine within tilda passive.
        states are:
        tilda_passive_states = {'idle': 0, 'scanning': 1, 'error': 2}
        :return number of the current state.
        """
        status_num = self.ReadWrite(TpCfg.TildaPassiveStateInd)
        return status_num

    def set_tilda_passive_status(self, status_num):
        """
        sets the status of the statemachine.
        states are: tilda_passive_states = {'idle': 0, 'scanning': 1, 'error': 2}
            idle can be left due to cmd in TpCfg.TildaPassiveStateCtrl
                either to scanning or error
            scanning can only be left by an timeout of the DMA Queue.
            error can only be left by command.

            -> bitfile must be reloaded for every scan!
            -> stop bitfile to stop acquisition.
        """
        self.ReadWrite(TpCfg.TildaPassiveStateCtrl, status_num)
        return self.checkFpgaStatus()

    def set_bin_num(self, bins_10ns):
        """
        Sets the number of bins that will, be acquired after one trigger.
        Width of one bin is one tick, so 10ns.
        :return: True if everything is fine, else warning
        """
        self.ReadWrite(TpCfg.nOfBins, bins_10ns)
        return self.checkFpgaStatus()

    def set_delay(self, delay_10ns):
        """
        set the delay/10ns relative to the falling edge in Dio24 of Controller 1
        (currently Ch24 on TTL-Linedriver 1)
        :return True if everything is fine, else warning
        """
        self.ReadWrite(TpCfg.delay_10ns_ticks, delay_10ns)
        return self.checkFpgaStatus()
