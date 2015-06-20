'''
Created on 20.06.2015

@author: skaufmann
'''

pipeData = {}


dummyScanParameters = {'MCSSelectTrigger': 0, 'delayticks': 0, 'nOfBins': 10000, 'nOfBunches': 1,
                               'measVoltPulseLength25ns': 400, 'measVoltTimeout10ns': 100,
                               'VoltOrScaler': False, 'stepSize': int('00000010000000000000', 2),
                               'start': int('00000000001000000000', 2), 'nOfSteps': 20,
                               'nOfScans': 5, 'invertScan': False,
                               'measureOffset': False, 'heinzingerControl': 1, 'heinzingerOffsetVolt':1000,
                               'waitForKepco25nsTicks': 40,
                               'waitAfterReset25nsTicks': 4000}


isotopeData = {'version': '0.1', 'type': 'trs', 'isotope': 'simontium_27',
               'nOfTracks': '1', 'colDirTrue': 'False', 'accVolt': '999.8',
               'laserFreq': '12568.73'}

programs = {
    'errorHandler': 0,
    'simpleCounter': 1,
    'continuousSequencer': 2,
    'dac': 3,
    }

pipeData.update(activeTrackPars=dummyScanParameters)
pipeData.update(activeIsotopePars=isotopeData)
pipeData.update(progConfigs=programs)

pipeData.update(curVoltInd=0)
pipeData.update(nOfTotalSteps=0)

print(pipeData)


