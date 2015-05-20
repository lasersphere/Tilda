"""

Created on '07.05.2015'

@author:'simkaufm'

"""
from Driver.DataAcquisitionFpga.TimeResolvedSequencer import TimeResolvedSequencer
from Service.Formating import Formatter
import polliPipe

import time
import numpy as np


class FpgaTest():
    def __init__(self):
        self.trs = TimeResolvedSequencer()
        self.form = Formatter()
        self.fullData = []


    def measureOneTrack(self, scanpars):
        self.trs.measureTrack(scanpars)
        while self.trs.getSeqState() == self.trs.TrsCfg.seqState['measureTrack']:
            result = self.trs.getData()
            if result['nOfEle'] == 0:
                break
            else:
                newdata = np.ctypeslib.as_array(result['newData'])
                # print(data)
                self.fullData.append(newdata)
                time.sleep(0.05)
        print(self.fullData)



maininst = FpgaTest()
print(maininst.measureOneTrack(maininst.trs.TrsCfg.dummyScanParameters))


