"""
Created on 20.06.2015

@author: skaufmann


Module containing the ScanParameters dictionaries as needed for Scanning with the standard Sequencers
"""

# import Service.Formating as form


draftIsotopePars = {'version': '1.06', 'type': 'cs', 'isotope': 'calcium_40',
               'nOfTracks': '1', 'accVolt': '9999.8',
               'laserFreq': '12568.766'}

draftTrackPars = {'dacStepSize18Bit': 10,  # form.get24BitInputForVoltage(1, False),
                   'dacStartRegister18Bit': 10,  # form.get24BitInputForVoltage(-5, False),
                   'nOfSteps': 20,
                   'nOfScans': 30, 'nOfCompletedSteps': 0, 'invertScan': False,
                   'postAccOffsetVoltControl': 2, 'postAccOffsetVolt': 1000,
                   'waitForKepco25nsTicks': 40,
                   'waitAfterReset25nsTicks': 4000,
                   'activePmtList': [0, 1],
                   'colDirTrue': 'False',
                   'dwellTime10ns': 2000000,
                  'workingTime': None
                   }

draftMeasureVoltPars = {'measVoltPulseLength25ns': 400, 'measVoltTimeout10ns': 100}

draftPipeInternals = {
    'curVoltInd': 0,
    'activeTrackNumber': 0,
    'filePath': 'D:\\XMLTests_150807',
    'activeXmlFilePath': None,
    'activeGraphicsWindow': None
}

draftScanDict = {'isotopeData': draftIsotopePars,
                 'activeTrackPar': draftTrackPars,
                 'pipeInternals': draftPipeInternals,
                 'measureVoltPars': draftMeasureVoltPars
                 }

# for the time resolved sequencer us this:
# draftTrackPars = {'MCSSelectTrigger': 0, 'delayticks': 0, 'nOfBins': 10000, 'nOfBunches': 1,
#                                'measVoltPulseLength25ns': 400, 'measVoltTimeout10ns': 100,
#                                'VoltOrScaler': False, 'dacStepSize18Bit': int('00000010000000000000', 2),
#                                'dacStartRegister18Bit': int('00000000001000000000', 2), 'nOfSteps': 20,
#                                'nOfScans': 30, 'nOfCompletedSteps': 0, 'invertScan': False,
#                                'measureOffset': False, 'postAccOffsetVoltControl': 1, 'postAccOffsetVolt': 1000,
#                                'waitForKepco25nsTicks': 40,
#                                'waitAfterReset25nsTicks': 4000,
#                                'activePmtList': [0, 2, 4]
#                                }