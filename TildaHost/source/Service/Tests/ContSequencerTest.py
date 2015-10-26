"""

Created on '09.07.2015'

@author:'simkaufm'

Module for testing the Continous Sequencer.

"""

import time
import logging
import sys

from Driver.DataAcquisitionFpga.ContinousSequencer import ContinousSequencer
from Service.Scan.draftScanParameters import draftScanDict
import Service.AnalysisAndDataHandling.tildaPipeline as TildaPipe
import Driver.DataAcquisitionFpga.ContinousSequencerConfig as CsCfg
import Service.Formating as form
from Service.VoltageConversions.VoltageConversions import get_18bit_from_voltage, get_18bit_stepsize


logging.basicConfig(level=getattr(logging, 'INFO'), format='%(message)s', stream=sys.stdout)


"""
get the pipeline ready and type your scanparameters in here:
"""
measState = CsCfg.seqStateDict['measureTrack']
scanPars = draftScanDict
scanPars['isotopeData']['isotope'] = 'Nothing'
scanPars['pipeInternals']['workingDirectory'] = 'D:\\PulserOfflineTests_150806'
scanPars['activeTrackPar']['dwellTime10ns'] = 2000000
scanPars['activeTrackPar']['dacStepSize18Bit'] = get_18bit_stepsize(0.02)
scanPars['activeTrackPar']['dacStartRegister18Bit'] = get_18bit_from_voltage(-10)
scanPars['activeTrackPar']['postAccOffsetVolt'] = 500
scanPars['activeTrackPar']['postAccOffsetVoltControl'] = 2
scanPars['activeTrackPar']['nOfSteps'] = 100
scanPars['activeTrackPar']['nOfScans'] = 400
pipe = TildaPipe.CsPipe(scanPars)
pipe.start()  #dacStartRegister18Bit the pipeLine

"""
dacStartRegister18Bit devices and measurement here:
"""
cs = ContinousSequencer()  # dacStartRegister18Bit the FPGA

# hz2 = hz.Heinzinger(hz.hzCfg.comportHeinzinger2)  # dacStartRegister18Bit the Offset Heinzinger. Only Hz2 available right now.
#
# hz2.setVoltage(scanPars['activeTrackPar']['postAccOffsetVolt'])
# logging.info('Heinzinger 2 is set to: ' + str(hz2.getVoltage()) + 'V')

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