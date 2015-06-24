"""

Created on '20.05.2015'

@author:'simkaufm'

"""

import numpy as np
import os.path

import pickle
from Service.AnalysisAndDataHandling.tildaPipeline import TrsPipe
from Driver.DataAcquisitionFpga.TimeResolvedSequencerConfig import TRSConfig
import Service.FolderAndFileHandling as filehand
import Service.draftScanParameters as draftScan


path = 'TildaHost\\source\\Scratch\\exampleTRSRawData.py'
file = os.path.join(filehand.findTildaFolder(), path)
trsExampleData = pickle.load(open(file, 'rb'))[1:]
# print(trsExampleData)
exampleCfg = TRSConfig()
pipe = TrsPipe(draftScan.draftTrackPars)

pipe.start()


for i,j in enumerate(trsExampleData):
    pipe.feed(j)
    pipe.save()
# pipe.save()

#print(scalerArray)
# for i,j in enumerate(np.argwhere(pipe.pipeData['scalerArray'])):
#    print(pipe.pipeData['scalerArray'][j[0]][j[1]], j)
# print((pipe.pipeData['scalerArray'][0:][0:]))
#for i,j in enumerate(np.argwhere(pipe.pipeData['scalerArray'][:][:][0])):
#    print(pipe.pipeData['scalerArray'][j[0]][j[1]], j)

