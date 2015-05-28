"""

Created on '20.05.2015'

@author:'simkaufm'

"""

import numpy as np
import os.path

import pickle
from Service.AnalysisAndDataHandling.tildaPipeline import tildapipe
from Driver.DataAcquisitionFpga.TimeResolvedSequencerConfig import TRSConfig
import Service.FolderAndFileHandling as filehand


findit = filehand.FindTildaFolder()
path = 'TildaHost\\source\\Scratch\\exampleTRSRawData.py'
file = os.path.join(findit, path)
trsExampleData = pickle.load(open(file, 'rb'))[1:]
# print(trsExampleData)
exampleCfg = TRSConfig()
pipe = tildapipe()
pipe.start()
pipe.pipeData.update(exampleCfg.dummyScanParameters) #load all scanparameters into pipeline
pipe.pipeData.update(curVoltInd=0)
voltArray = np.zeros(exampleCfg.dummyScanParameters['nOfSteps'], dtype=np.uint32)
timeArray = np.arange(exampleCfg.dummyScanParameters['delayticks']*10,
                      (exampleCfg.dummyScanParameters['delayticks']*10 + exampleCfg.dummyScanParameters['nOfBins']*10),
                      10, dtype=np.uint32)
scalerArray = np.zeros((exampleCfg.dummyScanParameters['nOfSteps'], exampleCfg.dummyScanParameters['nOfBins'], 8), dtype=np.uint32)
pipe.pipeData.update(voltArray=voltArray, timeArray=timeArray, scalerArray=scalerArray, nOfTotalSteps=0)
# print(len(scalerArray[0]))
for i,j in enumerate(trsExampleData):
    pipe.feed(j)
# for i,j in enumerate(np.argwhere(pipe.pipeData['scalerArray'])):
#    print(pipe.pipeData['scalerArray'][j[0]][j[1]], j)
# print((pipe.pipeData['scalerArray'][0:][0:]))
#for i,j in enumerate(np.argwhere(pipe.pipeData['scalerArray'][:][:][0])):
#    print(pipe.pipeData['scalerArray'][j[0]][j[1]], j)
print(pipe.pipeData['nOfTotalSteps'])
print(pipe.pipeData['voltArray'])