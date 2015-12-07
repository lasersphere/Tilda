"""
Created on 

@author: simkaufm

Module Description: Modukle for running and controling the SimpleCounter Bitfile on the FPGA
"""

from Driver.DataAcquisitionFpga.SimpleCounter import SimpleCounter
import Service.AnalysisAndDataHandling.tildaPipeline as Tp


class SimpleCounterControl:
    def __init__(self, act_pmt_list, datapoints):
        """
        module for reading from the simple counter.
        """
        self.sc_pipe = None
        self.sc_pipe = Tp.simple_counter_pipe()
        self.sc_pipe.pipeData = {'activePmtList': act_pmt_list,
                                 'plotPoints': datapoints}
        self.sc_pipe.start()
        self.sc = None
        # must start reading immediately because otherwise fpga overfills buffer

    def run(self):
        """
        start the simple counter bitfile on the fpga
        """
        self.sc = SimpleCounter()

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
        return self.sc.DeInitFpga()
