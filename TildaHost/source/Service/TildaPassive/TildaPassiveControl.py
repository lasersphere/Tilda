"""
Created on 12.04.16

@author: simkaufm

Module Description:

Module for controlling the execution of the TildaPassive "mode" or bitfile.

"""

from Driver.DataAcquisitionFpga.TildaPassive import TildaPassive
import Driver.DataAcquisitionFpga.TildaPassiveConfig as TpCfg
import Service.AnalysisAndDataHandling.tildaPipeline as Tp

import time
import logging


class TildaPassiveControl:
    def __init__(self):
        """
        Module for operating the passive mode of Tilda
        """
        self.tp_pipe = Tp.tilda_passive_pipe()
        self.tp_pipe.start()
        self.tp_inst = None

    def stop(self):
        """
        clear the pipeline and save all nodes by that.
        deinitialize the fpga
        """
        self.tp_pipe.clear()
        if self.tp_inst.type == 'tipa':
            self.tp_inst.DeInitFpga()
        self.tp_inst = None

    def run(self):
        """
        start the Tilda Passive Bitfile on the Fpga
        """
        self.tp_inst = TildaPassive()

    def start_scanning(self):
        """
        set the state of Tilda passive to scanning and check if state is changed.
        Be sure to set the delay and number of bins before!
        :return: True if success
        """
        status_num = TpCfg.tilda_passive_states.get('scanning')
        self.tp_inst.set_tilda_passive_status(status_num)
        tries = 0
        max_tries = 10
        while tries < max_tries:
            status_num_answ = self.tp_inst.read_tilda_passive_status()
            if status_num_answ == status_num:
                return True
            else:
                time.sleep(0.05)
                tries += 1
        return False

    def set_values(self, n_of_bins, delay_10ns):
        """
        will set the number if bins, with a width of 10ns each
        and the delay to the falling edge of DIO24 on controller 1.
        :return True if success
        """
        state = self.tp_inst.read_tilda_passive_status()
        if state == TpCfg.tilda_passive_states.get('idle'):
            bins_succes = self.tp_inst.set_bin_num(n_of_bins)
            delay_success = self.tp_inst.set_delay(delay_10ns)
            return bins_succes and delay_success
        else:
            logging.debug('could not set values, because state is not idle but: %s' % state)
            return False
