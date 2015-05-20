"""

Created on '07.05.2015'

@author:'simkaufm'

"""
from Driver.DataAcquisitionFpga.TimeResolvedSequencer import TimeResolvedSequencer
from Service.Formating import Formatter
import time
import numpy as np
import ctypes

class FpgaTest():
    def __init__(self):
        self.trs = TimeResolvedSequencer()
        self.form = Formatter()
        self.fullData = []

    def measureOneTrack(self, scanpars):
        self.trs.measureTrack(scanpars)
        while self.trs.getSeqState() == self.trs.TrsCfg.seqState['measureTrack']:
            result = self.trs.getData()
            # print('type of new Data: ' + str(type(result['newData'])))
            if result['nOfEle'] == 0:
                break
            else:
                data = result['newData']
                print(result)
                print('type of numpyArray: ' + str(type(np.ctypeslib.as_array(result['newData']))))
                print('just the data: ' + str(data))
                print('pointer on data: ' + str(ctypes.byref(data)))
                time.sleep(0.4)
        # print(self.fullData)



maininst = FpgaTest()
print(maininst.measureOneTrack(maininst.trs.TrsCfg.dummyScanParameters))


