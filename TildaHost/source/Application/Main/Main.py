"""

Created on '07.05.2015'

@author:'simkaufm'

"""

from Driver.DataAcquisitionFpga.TimeResolvedSequencer import TimeResolvedSequencer
from Service.Formating import Formatter
import time

class Main():
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
                print(result)
                newData = [self.form.integerSplitHeaderInfo(result['newData'][i]) for i in range(len(result['newData']))]
                print(newData)
                self.fullData.append(newData)
                time.sleep(0.2)
        print(self.fullData)



maininst = Main()
print(maininst.measureOneTrack(maininst.trs.TrsCfg.dummyScanParameters))

print(format(814743552, '032b'))
# print(len(bin(814743552)[2:]))