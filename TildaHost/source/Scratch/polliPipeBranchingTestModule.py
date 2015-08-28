"""

Created on '27.08.2015'

@author:'simkaufm'


This module was created to test the Pipeline towards branching with plots.

To get this going, please set the TildaHost/source folder AND the PolliFit/source folders as Sources Root directory
"""

import Service.AnalysisAndDataHandling.tildaPipeline as TP
import matplotlib.pyplot as plt

# example scan dictionary:
scnd = {
    'measureVoltPars': {'measVoltPulseLength25ns': 400, 'measVoltTimeout10ns': 100},
        'isotopeData': {'accVolt': 9999.8, 'type': 'cs', 'isotopeStartTime': '2015-08-27 14:45:18',
                        'version': 1.06, 'nOfTracks': 1, 'isotope': 'Ca_40', 'laserFreq': 12568.766},
        'pipeInternals': {'curVoltInd': 0, 'activeXmlFilePath': None, 'activeTrackNumber': 0,
                          'filePath': None},
        'activeTrackPar': {'dacStartRegister18Bit': 503312, 'invertScan': 0, 'waitForKepco25nsTicks': 40,
                           'nOfSteps': 61, 'waitAfterReset25nsTicks': 4000, 'dacStepSize18Bit': 520,
                           'activePmtList': [0, 1], 'nOfCompletedSteps': 0, 'postAccOffsetVolt': 500,
                           'dwellTime10ns': 2000000, 'nOfScans': 50, 'colDirTrue': False,
                           'workingTime': ['unknown'], 'postAccOffsetVoltControl': 2}
        }

exampleData = [815246864]
longer_example_data = [815246864, 545260388, 562037180, 578813952, 595591168, 612368384, 629145600,
645922816, 662700032, 815247384, 545260300, 562037175, 578813952, 595591168, 
612368384, 629145600, 645922816, 662700032, 815247904, 545260350, 562037218, 
578813952, 595591168, 612368384, 629145600, 645922816, 662700032, 815248424, 
545260351, 562037221, 578813952, 595591168, 612368384, 629145600, 645922816, 
662700032, 815248944, 545260353, 562037197, 578813952, 595591168, 612368384, 
629145600, 645922816, 662700032, 815249464, 545260377, 562037213, 578813952, 
595591168, 612368384, 629145600, 645922816, 662700032, 815249984, 545260388, 
562037191, 578813952, 595591168, 612368384, 629145600, 645922816, 662700032, 
815250504, 545260358, 562037208, 578813952, 595591168, 612368384, 629145600, 
645922816, 662700032, 815251024, 545260379, 562037207, 578813952, 595591168, 
612368384, 629145600, 645922816, 662700032, 815251544, 545260401, 562037176, 
578813952, 595591168, 612368384, 629145600, 645922816, 662700032, 815252064, 
545260421, 562037201, 578813952, 595591168, 612368384, 629145600, 645922816, 
662700032, 815252584]

def TestPipe():

    start = TP.Node()

    pipe = TP.Pipeline(start)
    pipe.pipeData = scnd
    fig, axes = plt.subplots(2, sharex=True)

    walk = start.attach(TP.TN.NSplit32bData())
    walk = walk.attach(TP.TN.NSortRawDatatoArray())

    # branching works for now by using shallow copy, if the NotImplementedError occures
    # this is realized by a try loop in processItem in node.py
    # if this is removed and deepcopy is used, the pipeline will fail
    branch = walk.attach(TP.TN.NAccumulateSingleScan())

    # this part is not even necessary to cause the NotImplementedError with deepcopy()
    # branch1 = branch.attach(TP.TN.NArithmetricScaler([0]))
    # branch1 = branch1.attach(TP.TN.NMPlLivePlot(axes[0], 'single Scan scaler 0'))

    walk = walk.attach(TP.TN.NRemoveTrackCompleteFlag())
    walk = walk.attach(TP.TN.NSumCS())

    walk = walk.attach(TP.TN.NMPlLivePlot(axes[1], 'live sum'))

    return pipe


pipe = TestPipe()
pipe.start()
import numpy
numpy.set_printoptions(threshold=numpy.nan)

pipe.feed(exampleData)
pipe.clear()