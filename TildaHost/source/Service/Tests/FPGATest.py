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

import time
import numpy as np


class FpgaTest():
    def __init__(self):
        self.trs = TimeResolvedSequencer()
        self.form = Formatter()
        start = Node()
        walk = start.attach(NrawFormatToReadable())
        walk = walk.attach(NPrint())

        self.pipe = Pipeline(start)
        self.pipe.start()


    def measureOneTrack(self, scanpars):
        self.trs.measureTrack(scanpars)
        while self.trs.getSeqState() == self.trs.TrsCfg.seqState['measureTrack']:
            result = self.trs.getData()
            if result['nOfEle'] == 0:
                break
            else:
                newdata = np.ctypeslib.as_array(result['newData'])
                self.pipe.feed(newdata)
                time.sleep(0.05)




maininst = FpgaTest()
print(maininst.measureOneTrack(maininst.trs.TrsCfg.dummyScanParameters))


