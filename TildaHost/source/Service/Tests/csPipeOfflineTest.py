__author__ = 'Simon-K'

"""
Module for working with .raw data and testing the Pipeline.
"""

import Service.FolderAndFileHandling as FileHandle
import Service.AnalysisAndDataHandling.tildaPipeline as TildaPipe
import Service.Formating as Form
import Service.draftScanParameters as Drafts


import logging
import os
import sys
import time


logging.basicConfig(level=getattr(logging, 'INFO'), format='%(message)s', stream=sys.stdout)

# path = 'R:\\Projekte\\TRIGA\\Measurements and Analysis_Simon\\TILDATest_15_07_29\\CalciumOfflineTests_150728\\sortedByRuns'
# workdir = 'D:\\TildaOfflinePipeTests'

path = 'C:\\Workspace\\TildaTestData\\TILDATest_15_07_29\\CalciumOfflineTests_150728\\sortedByRuns'
workdir = 'C:\\TildaOfflinePipeTests'
runList = [x[0] for x in os.walk(path)][1:]
rawfiles = [[os.path.join(pathOfRun, file) for file in os.listdir(pathOfRun) if file.endswith('.raw')]
         for pathOfRun in runList]
xmlFiles = [[os.path.join(pathOfRun, file) for file in os.listdir(pathOfRun) if file.endswith('.xml')]
         for pathOfRun in runList]
scandicts = [FileHandle.scanDictionaryFromXmlFile(xmlFile[0], 0, {}) for xmlFile in xmlFiles]
# # v104scandict = scandicts[0][0]
# # v106scandict = Form.convertScanDictV104toV106(v104scandict, Drafts.draftScanDict)
# scandicts[0][0]['isotopeData']['colDirTrue'] = 'True'
'''the scans has been collected with the sequencer of version 1.04, renaming etc. requires translation: '''
scandicts = [Form.convertScanDictV104toV106(scandicts[i][0], Drafts.draftScanDict) for i, j in enumerate(scandicts)]

scandicts = scandicts[:1]
# print(len(scandicts))

for i, k in enumerate(scandicts):
    runNumber = i
    activeScandict = scandicts[runNumber]
    activeScandict.pop('trackPars')  # drop the dictionary which contains all tracks, beacue the pipeline only needs teh active one
    activeScandict['pipeInternals']['filePath'] = workdir
    activeScandict['activeTrackPar']['nOfCompletedSteps'] = 0

    cspipe, plots = TildaPipe.CsPipe(activeScandict)
    cspipe.start()

    for file in rawfiles[runNumber]:
        # print(type(FileHandle.loadPickle(file)), ' type loaded file:', os.path.split(file)[1])
        cspipe.feed(FileHandle.loadPickle(file))
        time.sleep(0.05)
    # cspipe.clear(cspipe.pipeData)

# input('press anything to exit ')
time.sleep(1)