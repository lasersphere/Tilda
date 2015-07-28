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

logging.basicConfig(level=getattr(logging, 'DEBUG'), format='%(message)s', stream=sys.stdout)

# TildaPipe.CsPipe()

path = 'C:\\Workspace\\TildaTestData\\raw'
prun0 = os.path.join(path, 'run0')
filesrun0 = [file for file in os.listdir(prun0) if file.endswith('.raw')]
xmlFile = os.path.join(prun0, [file for file in os.listdir(prun0) if file.endswith('.xml')][0])
scandict = FileHandle.scanDictionaryFromXmlFile(xmlFile, 0, {})

scandict['pipeInternals']['filePath'] = os.path.split(scandict['pipeInternals']['filePath'])[0]
scandict['activeTrackPar']['nOfCompletedSteps'] = 0
#
cspipe = TildaPipe.CsPipe()
cspipe.start()
print(cspipe.pipeData)

cspipe.feed(scandict)
print(cspipe.pipeData)

# for file in filesrun0:
#     cspipe.feed(FileHandle.loadPickle(os.path.join(prun0, file)))
# cspipe.clear(cspipe.pipeData)

# prun4 = os.path.join(path, 'run4')
# filesrun4 = [file for file in os.listdir(prun4) if file.endswith('.raw')]
# for file in filesrun4:
#     cspipe.feed(FileHandle.loadPickle(os.path.join(prun4, file)))
# print(cspipe.pipeData)
