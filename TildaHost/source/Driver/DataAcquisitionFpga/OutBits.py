"""
Created on 13.07.2017

@author: simkaufm

Module Description:  Module for controlling the outbits in each sequencer.

"""

import logging
import numpy as np

from Driver.DataAcquisitionFpga.FPGAInterfaceHandling import FPGAInterfaceHandling


class Outbits(FPGAInterfaceHandling):
    """
    class for operating the OutBits in a sequencer.
    Control of the OutBits state machine is done by the sequencer itself,
    but the commands for the outbits are loaded to the fpga via DMA-FIFO.
    This FIFO will be filled from this class.
    """

    ''' reading '''

    def read_outbits_number_of_cmds(self):
        """
        will read the number of currently stored commands on the fpga memory.
        be careful:
            this number is only increased by the fpga as soon
            as the data is transferred from the DMA-FIFO to the block memory.
            This happens automatically a t the beginning of each scan.
            After each completed track, the memory is cleared again and the numbers will be 0
        :return: tuple, (nOfCommansOutBit0, nOfCommansOutBit1, nOfCommansOutBit2)
        """
        ret = (self.ReadWrite(self.config.nOfCmdsOutbit0),
               self.ReadWrite(self.config.nOfCmdsOutbit1),
               self.ReadWrite(self.config.nOfCmdsOutbit2))
        return ret

    def read_outbits_state(self):
        """
        This will return the state of the Outbit state machine
        :return: tuple, (state_num_int, state_name_str)
        """
        state_num = self.ReadWrite(self.config.OutBitsState).value
        states = {
            0: 'init',
            1: 'idle',
            2: 'loading',
            3: 'listening',
            4: 'clearing',
            5: 'error'
        }
        return state_num, states.get(state_num, 'unknown')

    ''' writing: '''

    def set_outbits_cmd(self, outbit_cmd_dict):
        """
        call this function to write the settings for the outbits to the fpga.
        control of setting the outbits is handled internally by the sequencer.

        :param outbit_cmd_dict: dict, looks like this:
            {
            'outbit0': [('toggle'/'on'/'off', 'step'/'scan', step/scan_number_int), …],
            'outbit1': [('toggle'/'on'/'off', 'step'/'scan', step/scan_number_int), …],
            'outbit2': [('toggle'/'on'/'off', 'step'/'scan', step/scan_number_int), …]
            }
        """
        try:
            cmd_32b_l, orig_cmd_l = self._convert_dict_of_cmd_to_32b_eles(outbit_cmd_dict)
        except Exception as e:
            logging.error('could not convert the outbit dictionary'
                          ' %s to 32b commands, error is: %s' % (outbit_cmd_dict, e))
            return False
        if len(cmd_32b_l):
            logging.info('writing %d outbit commands.' % len(cmd_32b_l))
            self._write_to_target(cmd_32b_l)
        return self.checkFpgaStatus()

    def _convert_dict_of_cmd_to_32b_eles(self, outbit_cmd_dict):
        """
        the human readable list needs to be converted to 32b elements
        Each 32b element looks like this:

            Bit 31	Bit 30	Bit 29	Bit 28	Bit 27	Bit 26	Bit 25	Bit 24	Bit 23 - 0
            bit2	bit1	bit0	toggle	on	    off	    step	scan	scan/step number 0- 2^24

        :parameter outbit_cmd_dict: dict, looks like this:
            {
            'outbit0': [('toggle'/'on'/'off', 'step'/'scan', step/scan_number_int), …],
            'outbit1': [('toggle'/'on'/'off', 'step'/'scan', step/scan_number_int), …],
            'outbit2': [('toggle'/'on'/'off', 'step'/'scan', step/scan_number_int), …]
            }
        :return: np.array of 32b elements to write to the DMA-FIFO
        """
        bits = {'outbit0': 2 ** 29, 'outbit1': 2 ** 30, 'outbit2': 2 ** 31}
        modes = {'toggle': 2 ** 28, 'on': 2 ** 27, 'off': 2 ** 26}
        scan_step_mode = {'step': 2 ** 25, 'scan': 2 ** 24}
        ret_arr = np.zeros(0, dtype=np.uint32)
        org_cmd_list = []
        for bit_name, cmd_list in outbit_cmd_dict.items():
            for cmd in cmd_list:
                bit_cmd = bits[bit_name] + modes[cmd[0]] + scan_step_mode[cmd[1]] + cmd[2]
                ret_arr = np.append(ret_arr, np.array([bit_cmd], dtype=np.uint32))
                org_cmd_list += [[bit_name, cmd]]
        return ret_arr, org_cmd_list

    def _write_to_target(self, data):
        """
        function to pass a set of commands to the ppg
        :param data: numpy array containing the commands
        :return:
        """
        still_free = self.WriteU32Fifo(self.config.OutbitsCMD['ref'], data)
        return still_free


if __name__ == '__main__':
    outbit = Outbits('', '', '', dummy=True)
    to_convert = {
        'outbit0': [('toggle', 'scan', 1)],
        'outbit1': [('on', 'step', 2), ('off', 'step', 5)],
        'outbit2': [('off', 'scan', 3), ('off', 'scan', 10)]
    }

    converted, org_l = outbit._convert_dict_of_cmd_to_32b_eles(to_convert)
    for i, each in enumerate(converted):
        print(format(each, '032b'), org_l[i])
    print(org_l)
