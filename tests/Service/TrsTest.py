"""

Created on '07.05.2015'

@author:'simkaufm'

"""
import time
import logging
import sys

from Tilda.Driver.DataAcquisitionFpga.TimeResolvedSequencer import TimeResolvedSequencer
import Tilda.Driver.DataAcquisitionFpga.TimeResolvedSequencerConfig as TrsCfg
from Tilda.Service.AnalysisAndDataHandling.tildaPipeline import TrsPipe
import Tilda.Service.Scan.draftScanParameters as draftScan

# import pickle

logging.basicConfig(level=getattr(logging, 'DEBUG'), format='%(message)s', stream= sys.stdout)

class FpgaTest():
    measState = TrsCfg.seqStateDict['measureTrack']

    def __init__(self):
        self.trs = TimeResolvedSequencer()
        self.finalData = []
        self.pipe = TrsPipe(draftScan.draftScanDict)

        self.pipe.start()


    def measureOneTrack(self, scanpars):
        self.trs.measureTrack(scanpars)
        state = self.measState
        timeout = 0
        while state == self.measState:
            state = self.trs.getSeqState(TrsCfg)
            result = self.trs.getData(TrsCfg)
            if result['nOfEle'] == 0:
                if state == self.measState and timeout < 100:
                    time.sleep(0.05)
                    timeout += 1
                else:
                    break
            else:
                newdata = result['newData']
                logging.debug('newData is: ' + str(newdata))
                self.pipe.feed(newdata)
                # print(self.finalData)
                time.sleep(0.05)
                timeout += 1
        print('total steps are: ', self.pipe.pipeData['track0']['nOfCompletedSteps'])
        # print(self.finalData, 'final Data')
        # use pickle to save these dummy data!
        # pickle.dump(self.finalData, open(self.outfile, 'wb'))


maininst = FpgaTest()
print(maininst.measureOneTrack(TrsCfg.dummyScanParameters))


