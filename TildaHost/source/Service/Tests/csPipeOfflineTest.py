__author__ = 'Simon-K'

"""
Module for working with .raw data and testing the Pipeline.
"""

import Service.FolderAndFileHandling as FileHandle
import Service.AnalysisAndDataHandling.tildaPipeline as TildaPipe
import Service.Formating as Form

import logging
import os
import sys


logging.basicConfig(level=getattr(logging, 'INFO'), format='%(message)s', stream=sys.stdout)

path = 'R:\\Projekte\\TRIGA\\Measurements and Analysis_Simon\\TildaTestData\\sortedForOfflineTests'
runList = [x[0] for x in os.walk(path)][1:]
rawfiles = [[os.path.join(pathOfRun, file) for file in os.listdir(pathOfRun) if file.endswith('.raw')]
         for pathOfRun in runList]
xmlFiles = [[os.path.join(pathOfRun, file) for file in os.listdir(pathOfRun) if file.endswith('.xml')]
         for pathOfRun in runList]
scandicts = [FileHandle.scanDictionaryFromXmlFile(xmlFile[0], 0, {}) for xmlFile in xmlFiles]

runNumber = 1
activeScandict = scandicts[runNumber]
activeScandict['pipeInternals']['filePath'] = 'D:\\TildaOfflinePipeTests'
activeScandict['activeTrackPar']['nOfCompletedSteps'] = 0
# print(activeScandict['pipeInternals']['filePath'])
# #
cspipe = TildaPipe.CsPipe(activeScandict)
cspipe.start()
# print(cspipe.pipeData)
# #
# cspipe.feed(scandict)
# print(cspipe.pipeData)

for file in rawfiles[runNumber]:
    # print(FileHandle.loadPickle(file))
    cspipe.feed(FileHandle.loadPickle(file))
# cspipe.feed(0)
cspipe.clear(cspipe.pipeData)

# prun4 = os.path.join(path, 'run4')
# filesrun4 = [file for file in os.listdir(prun4) if file.endswith('.raw')]
# for file in filesrun4:
#     cspipe.feed(FileHandle.loadPickle(os.path.join(prun4, file)))
# print(cspipe.pipeData)


