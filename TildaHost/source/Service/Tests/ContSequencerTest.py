"""

Created on '09.07.2015'

@author:'simkaufm'

"""

from Driver.DataAcquisitionFpga.ContinousSequencer import ContinousSequencer
from Service.draftScanParameters import draftScanDict
import Service.AnalysisAndDataHandling.tildaPipeline as TildaPipe
import Driver.DataAcquisitionFpga.ContinousSequencerConfig as CsCfg
import time

import logging
import sys

logging.basicConfig(level=getattr(logging, 'INFO'), format='%(message)s', stream= sys.stdout)


measState = CsCfg.seqStateDict['measureTrack']
scanPars = draftScanDict
scanPars['activeTrackPar']['dwellTime'] = 2000000
pipe = TildaPipe.CsPipe(scanPars)
pipe.start()
cs = ContinousSequencer()


def meaureOneTrack(scanparsDict):
    state = None
    result = {'nOfEle': None}
    timeout = 0
    if cs.measureTrack(scanparsDict):
        state = measState
    while state == measState or result['nOfEle'] > 0:
        state = cs.getSeqState(CsCfg)
        result = cs.getData(CsCfg)
        if result['nOfEle'] == 0:
            if state == measState and timeout < 100:
                time.sleep(0.05)
                timeout += 1
            else:
                break
        else:
            newdata = result['newData']
            pipe.feed(newdata)
            # print(self.finalData)
            time.sleep(0.05)
            timeout += 1
    pipe.clear(pipe.pipeData)


meaureOneTrack(draftScanDict)