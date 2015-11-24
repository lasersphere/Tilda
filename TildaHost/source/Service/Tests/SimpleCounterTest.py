"""
Created on 

@author: simkaufm

Module Description:
"""

from Driver.DataAcquisitionFpga.SimpleCounter import SimpleCounter

spc = SimpleCounter()

while True:
    print(spc.read_data_from_fifo())