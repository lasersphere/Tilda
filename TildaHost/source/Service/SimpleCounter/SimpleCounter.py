"""
Created on 

@author: simkaufm

Module Description: Module for running and controling the SimpleCounter Bitfile on the FPGA
"""

import Service.AnalysisAndDataHandling.tildaPipeline as Tp
from Driver.DataAcquisitionFpga.SimpleCounter import SimpleCounter
from Driver.DataAcquisitionFpga.SimpleCounterDummy import SimpleCounterDummy


class SimpleCounterControl:
    def __init__(self, act_pmt_list, datapoints, callback_sig, sample_interval):
        """
        module for reading from the simple counter.
        """
        self.sc_pipe = None
        self.sc_pipe = Tp.simple_counter_pipe(callback_sig, act_pmt_list, sample_interval)
        self.sc_pipe.pipeData = {'activePmtList': act_pmt_list,
                                 'plotPoints': datapoints}
        self.sc_pipe.start()
        self.sc = None
        # must start reading immediately because otherwise fpga overfills buffer

    def run(self):
        """
        start the simple counter bitfile on the fpga
        """
        try:
            self.sc = SimpleCounter()  # does not go automatically to dummy mode if hardware is not installed.
            return True
        except Exception as e:
            print('error: could not start simple counter, error is: %s ' % e)
            return False

    def run_dummy(self):
        """
        dummy simple counter
        """
        self.sc = SimpleCounterDummy()
        return True

    def read_data(self):
        """
        reads all data currently available from the fpga and feeds it to the pipeline.
        """
        data = self.sc.read_data_from_fifo()
        self.sc_pipe.feed(data['newData'])

    def stop(self):
        """
        deinitialize the fpga
        :return: status of fpga
        """
        self.sc_pipe.clear()
        fpga_status = True
        if self.sc.type == 'sc':
            fpga_status = self.sc.DeInitFpga()
        self.sc = None
        return fpga_status

    def set_post_acc_control(self, state_name):
        self.sc.set_post_acc_control(state_name)

    def get_post_acc_control(self):
        """
        :return: post_acc_state, post_acc_name
        """
        return self.sc.get_post_acc_control()

    def set_dac_volt(self, volt_dbl):
        """
        sets the voltage to the dac, input is double/float
        :return: status of fpga
        """
        return self.sc.set_dac_voltage(volt_dbl)
