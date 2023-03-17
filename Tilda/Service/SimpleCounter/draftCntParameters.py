"""
Created on 2022-01-17

@author: lrenth

Module containing the SimpleCounter parameters as needed for for "no-trigger" counting
"""

from Driver.DataAcquisitionFpga.TriggerTypes import TriggerTypes as TiTs

""" outer most dictionary contains the following keys: """
cntDict_list = ['cntData', 'timing']

''' The cntData dictionary is used for the whole measurement and contains the following keys: '''
cntData_list = ['type']

''' The timing dictionary sets up the timing for the measurement and contains the following keys: '''
timing_list = ['trigger', 'nOfBins']

''' The trigger dictionary sets up the trigger and contains the following keys: '''
trigger_list = ['meas_trigger']


''' some example values for the simple counter'''

draftCntDataPars = {'type': 'smplCnt'}

draftTriggerPars = {'meas_trigger': {'type': getattr(TiTs, 'no_trigger')}}

draftTimingPars = {
    'trigger': draftTriggerPars,
    'nOfBins': 1000
}

draftCntDict = {'cntData': draftCntDataPars,
                'timing': draftTimingPars}