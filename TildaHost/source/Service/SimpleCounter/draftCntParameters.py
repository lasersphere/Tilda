"""
Created on 2022-01-17

@author: lrenth

Module containing the SimpleCounter parameters as needed for for "no-trigger" counting
"""

from Driver.DataAcquisitionFpga.TriggerTypes import TriggerTypes as TiTs


draftCntPars = {
    'trigger': {
        'meas_trigger': {
            'type': TiTs.no_trigger
        }
    },
    'nOfBins': 1000,
}