"""

Created on '07.05.2015'

@author:'simkaufm'

"""
from Driver.DataAcquisitionFpga.TimeResolvedSequencer import TimeResolvedSequencer
from Service.Formating import Formatter
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
            # print('type of new Data: ' + str(type(result['newData'])))
            if result['nOfEle'] == 0:
                break
            else:
                print(result)
                print('type of numpyArray: ' + str(type(np.ctypeslib.as_array(result['newData']))))
                newData = [self.form.integerSplitHeaderInfo(np.ctypeslib.as_array(result['newData'])[i]) for i in range(len(result['newData']))]
                # print(newData)
                self.fullData.append(result['newData'])
                time.sleep(0.4)
        print(self.fullData)



maininst = FpgaTest()
print(maininst.measureOneTrack(maininst.trs.TrsCfg.dummyScanParameters))


