"""

Created on '07.05.2015'

@author:'simkaufm'

"""
from Driver.DataAcquisitionFpga.TimeResolvedSequencer import TimeResolvedSequencer
import Driver.DataAcquisitionFpga.TimeResolvedSequencerConfig as TrsCfg

import time
# import pickle

class FpgaTest():
    def __init__(self):
        self.outfile = 'D:\\Workspace\\PyCharm\\Tilda\\TildaHost\\source\\Scratch\\exampleTRSRawData.py'
        self.trs = TimeResolvedSequencer()
        self.finalData = [0]


    def measureOneTrack(self, scanpars):
        self.trs.measureTrack(scanpars)
        while self.trs.getSeqState(TrsCfg) == TrsCfg.seqStateDict['measureTrack']:
            result = self.trs.getData()
            if result['nOfEle'] == 0:
                break
            else:
                newdata = result['newData']
                self.finalData.append(newdata)
                # print(self.finalData)
                time.sleep(0.05)
        print(self.finalData)
        # use pickle to save these dummy data!
        # pickle.dump(self.finalData, open(self.outfile, 'wb'))



maininst = FpgaTest()
print(maininst.measureOneTrack(TrsCfg.dummyScanParameters))


