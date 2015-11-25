"""
Created on 

@author: simkaufm

Module Description:
"""

from Driver.DataAcquisitionFpga.SimpleCounter import SimpleCounter
from Service.AnalysisAndDataHandling.tildaPipeline import simple_counter_pipe as ScPipe

import time

sc = SimpleCounter()
time.sleep(1)

sc_pipe = ScPipe()
sc_pipe.pipeData = {'activePmtList': [0, 1], 'plotPoints': 60}
sc_pipe.start()

while True:
    data = sc.read_data_from_fifo()
    sc_pipe.feed(data['newData'])
    time.sleep(1)