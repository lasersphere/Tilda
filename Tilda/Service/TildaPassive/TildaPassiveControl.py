"""
Created on 12.04.16

@author: simkaufm

Module Description:

Module for controlling the execution of the TildaPassive "mode" or bitfile.

"""

from Tilda.Driver.DataAcquisitionFpga.TildaPassive import TildaPassive
from Tilda.Driver.DataAcquisitionFpga.TildaPassiveDummy import TildaPassiveDummy
import Tilda.Driver.DataAcquisitionFpga.TildaPassiveConfig as TpCfg
import Tilda.Service.AnalysisAndDataHandling.tildaPipeline as Tp

import time
import logging


class TildaPassiveControl:
    def __init__(self):
        """
        Module for operating the passive mode of Tilda
        """
        self.tp_pipe = None
        self.tp_inst = None  # instance of the loaded bitfile
        self.scan_pars = None

    def setup_tipa_ctrl(self, scan_pars, raw_callback, steps_scans_callback):
        """
        setup the pipeline and run the Bitfile.
        """
        self.scan_pars = scan_pars
        self.tp_pipe = Tp.tilda_passive_pipe(self.scan_pars, raw_callback, steps_scans_callback)
        self.tp_pipe.start()
        try:
            self.run()
        except Exception as e:
            logging.error('hardware not found, starting Tilda Passive dummy')
            self.run_dummy()

    def stop(self):
        """
        clear the pipeline and save all nodes by that.
        deinitialize the fpga
        """
        self.tp_pipe.stop()
        self.tp_pipe.clear()
        if self.tp_inst.type == 'tipa':
            self.tp_inst.DeInitFpga()
        self.tp_inst = None
        self.scan_pars = None

    def run(self):
        """
        start the Tilda Passive Bitfile on the Fpga
        """
        self.tp_inst = TildaPassive()

    def run_dummy(self):
        """
        start the dummmy if the hardware is not available
        """
        self.tp_inst = TildaPassiveDummy()

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
            status_num_answ = self.read_tipa_status()
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
        state = self.read_tipa_status()
        if state == TpCfg.tilda_passive_states.get('idle'):
            bins_succes = self.tp_inst.set_bin_num(n_of_bins)
            delay_success = self.tp_inst.set_delay(delay_10ns)
            return bins_succes and delay_success
        else:
            logging.debug('could not set values, because state is not idle but: %s' % state)
            return False

    def read_data(self):
        """
        function to read the data from the fpga and feed it directly into the pipeline
        """
        data = self.tp_inst.read_data_from_fifo()
        if data['nOfEle'] != 0:
            self.tp_pipe.feed(data['newData'])
            return True
        else:
            return False

    def read_tipa_status(self):
        """
        function for reading the status.
        tilda_passive_states = {'idle': 0, 'scanning': 1, 'error': 2}
        :return number of the current state.
        """
        return self.tp_inst.read_tilda_passive_status()

    def tipa_get_default_scan_pars(self):
        """
        returns the draft scan dictionary from the TildaPassiveConfig
        """
        return TpCfg.draft_tipa_scan_pars
