'''
Created on 20.06.2015

@author: skaufmann
'''

draftIsotopePars = {'version': '0.1', 'type': 'cs', 'isotope': 'simontium_27',
               'nOfTracks': '1', 'colDirTrue': 'False', 'accVolt': '9999.8',
               'laserFreq': '12568.73'}

draftTrackPars = {'MCSSelectTrigger': 0, 'delayticks': 0, 'nOfBins': 10000, 'nOfBunches': 1,
                               'measVoltPulseLength25ns': 400, 'measVoltTimeout10ns': 100,
                               'VoltOrScaler': False, 'stepSize': int('00000010000000000000', 2),
                               'start': int('00000000001000000000', 2), 'nOfSteps': 20,
                               'nOfScans': 30, 'nOfCompletedSteps': 0, 'invertScan': False,
                               'measureOffset': False, 'heinzingerControl': 1, 'heinzingerOffsetVolt': 1000,
                               'waitForKepco25nsTicks': 40,
                               'waitAfterReset25nsTicks': 4000,
                               'activePmtList': [0, 2, 4]
                               }

draftPipeInternals = {
    'curVoltInd': 0,
    'activeTrackNumber': 0,
    'filePath': 'D:\\Workspace\\Testdata',
    'activeXmlFilePath': None
}

draftScanDict = {'isotopeData': draftIsotopePars,
                 'activeTrackPar': draftTrackPars,
                 'pipeInternals': draftPipeInternals}