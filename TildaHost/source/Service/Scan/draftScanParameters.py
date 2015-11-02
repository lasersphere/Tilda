"""
Created on 20.06.2015

@author: skaufmann


Module containing the ScanParameters dictionaries as needed for Scanning with the standard Sequencers
"""
""" List of currently supported sequencer types """
sequencer_types_list = ['cs', 'trs']

""" outer most dictionary contains the following keys: """
scanDict_list = ['isotopeData', 'activeTrackPar', 'pipeInternals', 'measureVoltPars']

""" the isotopeData dictionary is used for the whole isotope and contains the following keys: """
isotopeData_list = ['version', 'type', 'isotope', 'nOfTracks', 'accVolt', 'laserFreq']

""" the pipeInternals dictionary is used by the pipeline and is in first place only valid for one track.
It contains the following keys: """
pipeInternals_list = ['curVoltInd', 'activeTrackNumber', 'workingDirectory', 'activeXmlFilePath']

""" the measureVoltPars dictionary is used to define the pulse length to start the voltage measurement
 and the timeout of the voltage measurement. It contains the following keys: """
measureVoltPars_list = ['measVoltPulseLength25ns', 'measVoltTimeout10ns']

""" the activeTrackPar dictionary is only used by one track but therefore beholds
the most information for the sequencer. It contains the following keys and
MUST be appended with the keys from the corresponding sequencer (see below): """
activeTrackPar_list = ['dacStepSize18Bit', 'dacStartRegister18Bit', 'nOfSteps', 'nOfScans', 'nOfCompletedSteps',
                       'invertScan', 'postAccOffsetVoltControl', 'postAccOffsetVolt', 'waitForKepco25nsTicks',
                       'waitAfterReset25nsTicks', 'activePmtList', 'colDirTrue', 'workingTime']

"""  each sequencer needs its own parameters and therefore, the keys are listed below
naming convention is type_list.  """
cs_list = ['dwellTime10ns']

trs_list = ['MCSSelectTrigger', 'delayticks', 'nOfBins', 'nOfBunches']

""" below are some example values which can be used for scanning: """
draftIsotopePars = {
    'version': '1.06', 'type': 'cs', 'isotope': '44Ca',
    'nOfTracks': '1', 'accVolt': '9999.8',
    'laserFreq': '12568.766'
}

draftTrackPars = {
    'dacStepSize18Bit': 1,  # form.get_24bit_input_from_voltage(1, False),
    'dacStartRegister18Bit': 0,  # form.get_24bit_input_from_voltage(-5, False),
    'nOfSteps': 20,
    'nOfScans': 30, 'nOfCompletedSteps': 0, 'invertScan': False,
    'postAccOffsetVoltControl': 2, 'postAccOffsetVolt': 1000,
    'waitForKepco25nsTicks': 40,
    'waitAfterReset25nsTicks': 4000,
    'activePmtList': [0, 1],
    'colDirTrue': False,
    'dwellTime10ns': 2000000,
    'workingTime': None
}

draftMeasureVoltPars = {'measVoltPulseLength25ns': 400, 'measVoltTimeout10ns': 100}

draftPipeInternals = {
    'curVoltInd': 0,
    'activeTrackNumber': 0,
    'workingDirectory': None,
    'activeXmlFilePath': None
}

draftScanDict = {'isotopeData': draftIsotopePars,
                 'activeTrackPar': draftTrackPars,
                 'pipeInternals': draftPipeInternals,
                 'measureVoltPars': draftMeasureVoltPars
                 }

# for the time resolved sequencer us this:
# draftTrackPars = {'MCSSelectTrigger': 0, 'delayticks': 0, 'nOfBins': 10000, 'nOfBunches': 1,
# 'measVoltPulseLength25ns': 400, 'measVoltTimeout10ns': 100,
# 'VoltOrScaler': False, 'dacStepSize18Bit': int('00000010000000000000', 2),
# 'dacStartRegister18Bit': int('00000000001000000000', 2), 'nOfSteps': 20,
# 'nOfScans': 30, 'nOfCompletedSteps': 0, 'invertScan': False,
# 'measureOffset': False, 'postAccOffsetVoltControl': 1, 'postAccOffsetVolt': 1000,
# 'waitForKepco25nsTicks': 40,
# 'waitAfterReset25nsTicks': 4000,
# 'activePmtList': [0, 2, 4]
# }