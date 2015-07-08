"""

Created on '07.05.2015'

@author:'simkaufm'

"""
from Driver.DataAcquisitionFpga.TimeResolvedSequencer import TimeResolvedSequencer
from Service.Formating import Formatter
from polliPipe.node import Node
from polliPipe.pipeline import Pipeline
from Service.AnalysisAndDataHandling.tildaNodes import NrawFormatToReadable
from polliPipe.simpleNodes import NPrint
import Driver.DataAcquisitionFpga.TimeResolvedSequencerConfig as TrsCfg

import time
# import pickle

class FpgaTest():
    def __init__(self):
        self.outfile = 'D:\\Workspace\\PyCharm\\Tilda\\TildaHost\\source\\Scratch\\exampleTRSRawData.py'
        self.trs = TimeResolvedSequencer()
        self.form = Formatter()
        start = Node()
        walk = start.attach(NPrint())
        # walk = walk.attach(NPrint())
        self.finalData = [0]

        self.pipe = Pipeline(start)
        self.pipe.start()


    def measureOneTrack(self, scanpars):
        self.trs.measureTrack(scanpars)
        while self.trs.getSeqState(TrsCfg) == TrsCfg.seqState['measureTrack']:
            result = self.trs.getData()
            if result['nOfEle'] == 0:
                break
            else:
                newdata = result['newData']
                self.finalData.append(newdata)
                # print(self.finalData)
                time.sleep(0.05)
        # print(self.finalData)
        # use pickle to save these dummy data!
        # pickle.dump(self.finalData, open(self.outfile, 'wb'))



maininst = FpgaTest()
print(maininst.measureOneTrack(TrsCfg.dummyScanParameters))


