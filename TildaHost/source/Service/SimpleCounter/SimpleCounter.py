"""
Created on 

@author: simkaufm

Module Description: Modukle for running and controling the SimpleCounter Bitfile on the FPGA
"""

from Driver.DataAcquisitionFpga.SimpleCounter import SimpleCounter
import Service.AnalysisAndDataHandling.tildaPipeline as Tp

import time
import multiprocessing
import queue


class SimpleCounterControl(SimpleCounter, multiprocessing.Process):
    def __init__(self, pipedat, cmd_queue):
        """
        module for reading from the simple counter.
        breaks as soon as something is send trhough the pipeline.
        :parameter: pipedat = {'activePmtList': active_pmt_list, 'plotPoints': plotpoints}
        """
        SimpleCounter.__init__(self)
        multiprocessing.Process.__init__(self)
        self.sc_pipe = None
        self.cmd_queue = cmd_queue
        self.pipe_dat = pipedat
        self.start()
        # must start reading immediately because otherwise fpga overfills buffer

    def run(self):
        proc_name = self.name
        self.sc_pipe = Tp.simple_counter_pipe()
        self.sc_pipe.pipeData = self.pipe_dat
        self.sc_pipe.start()
        while True:
            try:
                next_task = self.cmd_queue.get(block=False)
                print(proc_name, ' Exiting')
                self.DeInitFpga()
                break  # break if something was send through the cmd_queue
            except queue.Empty:  # i want an empty queue
                data = self.read_data_from_fifo()
                self.sc_pipe.feed(data['newData'])
                time.sleep(0.5)
