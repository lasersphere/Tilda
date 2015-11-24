"""
Created on 

@author: simkaufm

Module Description:
"""

from Driver.DataAcquisitionFpga.SimpleCounter import SimpleCounter

import time

spc = SimpleCounter()
time.sleep(1)

while True:
    print(spc.read_data_from_fifo())
    time.sleep(1)