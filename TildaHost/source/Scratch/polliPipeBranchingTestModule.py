"""

Created on '27.08.2015'

@author:'simkaufm'


This module was created to test the Pipeline towards branching with plots.

To get this going, please set the TildaHost/source folder AND the PolliFit/source folders as Sources Root directory
"""

import Service.AnalysisAndDataHandling.tildaPipeline as TP
import matplotlib.pyplot as plt
import numpy as np

# example scan dictionary:
scnd = {
    'measureVoltPars': {'measVoltPulseLength25ns': 400, 'measVoltTimeout10ns': 100},
        'isotopeData': {'accVolt': 9999.8, 'type': 'cs', 'isotopeStartTime': '2015-08-27 14:45:18',
                        'version': 1.06, 'nOfTracks': 1, 'isotope': 'Ca_40', 'laserFreq': 12568.766},
        'pipeInternals': {'curVoltInd': 0, 'activeXmlFilePath': None, 'activeTrackNumber': 0,
                          'workingDirectory': None},
        'track0': {'dacStartRegister18Bit': 503312, 'invertScan': 0, 'waitForKepco25nsTicks': 40,
                           'nOfSteps': 61, 'waitAfterReset25nsTicks': 4000, 'dacStepSize18Bit': 520,
                           'activePmtList': [0, 1], 'nOfCompletedSteps': 0, 'postAccOffsetVolt': 500,
                           'dwellTime10ns': 2000000, 'nOfScans': 50, 'colDirTrue': False,
                           'workingTime': ['unknown'], 'postAccOffsetVoltControl': 2}
        }



exampleData = np.random.random((scnd['track0']['nOfSteps'], len(scnd['track0']['activePmtList'])))

def TestPipe():

    start = TP.Node()

    pipe = TP.Pipeline(start)
    pipe.pipeData = scnd
    fig, axes = plt.subplots(2, sharex=True)

    walk = start.attach(TP.SN.NPrint())

    branch = walk.attach(TP.SN.NPrint())

    walk = walk.attach(TP.TN.NMPlLivePlot(axes[0], 'live sum'))

    return pipe


pipe = TestPipe()
pipe.start()
import numpy
numpy.set_printoptions(threshold=numpy.nan)

pipe.feed(exampleData)
pipe.clear()