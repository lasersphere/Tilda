from Tilda.PolliFit import TildaTools

__author__ = 'Simon-K'

"""
Module for working with .raw data and testing the Pipeline.
"""

import logging
import os
import time
import sys

import Tilda.Service.FileOperations.FolderAndFileHandling as FileHandle
import Tilda.Service.AnalysisAndDataHandling.tildaPipeline as TildaPipe
import Tilda.Service.Formatting as Form

# sys.ps1 = 'Necessary for matplotlib 1.4.0 to work the way it used to'

logging.basicConfig(level=getattr(logging, 'DEBUG'), format='%(message)s', stream=sys.stdout)
#
path = 'R:\\Projekte\\TRIGA\\Measurements and Analysis_Simon\\TILDATest_15_07_29\\CalciumOfflineTests_150728\\sortedByRuns'
workdir = 'D:\\TildaOfflinePipeTests'

# path = 'C:\\Workspace\\TildaTestData\\TILDATest_15_07_29\\CalciumOfflineTests_150728\\sortedByRuns'
# workdir = 'C:\\TildaOfflinePipeTests'
runList = [x[0] for x in os.walk(path)][1:]
rawfiles = [[os.path.join(pathOfRun, file) for file in os.listdir(pathOfRun) if file.endswith('.raw')]
         for pathOfRun in runList]
xmlFiles = [[os.path.join(pathOfRun, file) for file in os.listdir(pathOfRun) if file.endswith('.xml')]
         for pathOfRun in runList]
scandicts = [TildaTools.scan_dict_from_xml_file(xmlFile[0], 0, {}) for xmlFile in xmlFiles]
# # v104scandict = scandicts[0][0]
# # v106scandict = Form.convert_scandict_v104_to_v106(v104scandict, Drafts.draftScanDict)
# scandicts[0][0]['isotopeData']['colDirTrue'] = 'True'
'''the scans has been collected with the sequencer of version 1.04, renaming etc. requires translation: '''
scandicts = [Form.convert_scandict_v104_to_v106(scandicts[i][0]) for i, j in enumerate(scandicts)]

scandicts = scandicts[:1]
# print(len(scandicts))

# print(FileHandle.loadPickle(rawfiles[0][-1]))

for i, k in enumerate(scandicts):
    runNumber = i
    activeScandict = scandicts[runNumber]
    activeScandict.pop('trackPars')  # drop the dictionary which contains all tracks, beacue the pipeline only needs teh active one
    activeScandict['pipeInternals']['workingDirectory'] = workdir
    activeScandict['track0']['nOfCompletedSteps'] = 0

    cspipe = TildaPipe.CsPipe(activeScandict)
    cspipe.start()

    for file in rawfiles[runNumber]:
        # print(type(FileHandle.loadPickle(file)), ' type loaded file:', os.path.split(file)[1])
        cspipe.feed(FileHandle.loadPickle(file))
        time.sleep(0.05)
    cspipe.clear()
