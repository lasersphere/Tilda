"""

Created on '09.07.2015'

@author:'simkaufm'

"""

from Driver.DataAcquisitionFpga.ContinousSequencer import ContinousSequencer
from Service.draftScanParameters import draftScanDict
import Driver.DataAcquisitionFpga.ContinousSequencerConfig as CsCfg
import time

measState = CsCfg.seqStateDict['measureTrack']
pipe = 'blub'

def ms2ticks(val):
    """
    1 tick is 10 ns
    """
    return val*100000

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


cs = ContinousSequencer()
draftScanDict['activeTrackPar']['dwellTime'] = ms2ticks(20)






