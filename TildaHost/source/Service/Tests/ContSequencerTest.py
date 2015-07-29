"""

Created on '09.07.2015'

@author:'simkaufm'

Module for testing the Continous Sequencer.

"""

from Driver.DataAcquisitionFpga.ContinousSequencer import ContinousSequencer
from Service.draftScanParameters import draftScanDict
import Service.AnalysisAndDataHandling.tildaPipeline as TildaPipe
import Driver.DataAcquisitionFpga.ContinousSequencerConfig as CsCfg
import Service.Formating as form
import Driver.Heinzinger.Heinzinger as hz


import time
import logging
import sys

logging.basicConfig(level=getattr(logging, 'DEBUG'), format='%(message)s', stream=sys.stdout)


"""
get the pipeline ready and type your scanparameters in here:
"""
measState = CsCfg.seqStateDict['measureTrack']
scanPars = draftScanDict
scanPars['isotopeData']['isotope'] = 'Ca_40'
scanPars['pipeInternals']['filePath'] = 'D:\\CalciumOfflineTests_150728'
scanPars['activeTrackPar']['dwellTime'] = 2000000
scanPars['activeTrackPar']['stepSize'] = form.get24BitInputForVoltage(0.005, False, True)
scanPars['activeTrackPar']['start'] = form.get24BitInputForVoltage(-0.25, False)
scanPars['activeTrackPar']['heinzingerOffsetVolt'] = 500
scanPars['activeTrackPar']['nOfSteps'] = 45
scanPars['activeTrackPar']['nOfScans'] = 50
pipe = TildaPipe.CsPipe(scanPars)
pipe.start()  #start the pipeLine

"""
start devices and measurement here:
"""
cs = ContinousSequencer()  # start the FPGA

hz2 = hz.Heinzinger(hz.hzCfg.comportHeinzinger2)  # start the Offset Heinzinger. Only Hz2 available right now.

hz2.setVoltage(scanPars['activeTrackPar']['heinzingerOffsetVolt'])
logging.info('Heinzinger 2 is set to: ' + str(hz2.getVoltage()) + 'V')

def meaureOneTrack(scanparsDict):
    """
    function for the measurement of one complete track. will block the whole python execution while running.
    purpose is only for prototype testing.
    """
    state = None
    result = {'nOfEle': None}
    timeout = 0
    if cs.measureTrack(scanparsDict):
        state = measState
    while state == measState or result['nOfEle'] > 0:
        state = cs.getSeqState(CsCfg)
        result = cs.getData(CsCfg)
        if result['nOfEle'] == 0:
            if state == measState and timeout < 500:
                time.sleep(0.05)
                timeout += 1
            else:
                break
        else:
            newdata = result['newData']
            pipe.feed(newdata)
            # print(self.finalData)
            time.sleep(0.05)
            # timeout += 1
    logging.info('state is: ' + str(state) + ' timeout is: ' + str(timeout)
                 + ' Completed Steps: ' + str(pipe.pipeData['activeTrackPar']['nOfCompletedSteps']))
    pipe.clear(pipe.pipeData)

logging.info('starting measurement...')
meaureOneTrack(draftScanDict)