"""

Created on '20.05.2015'

@author:'simkaufm'

"""

import numpy as np

import pickle
from Service.AnalysisAndDataHandling.tildaPipeline import tildapipe
from Driver.DataAcquisitionFpga.TimeResolvedSequencerConfig import TRSConfig



file = 'D:\\Workspace\\PyCharm\\Tilda\\TildaHost\\source\\Scratch\\exampleTRSRawData.py'
trsExampleData = pickle.load(open(file, 'rb'))[1:]
# print(trsExampleData)
exampleCfg = TRSConfig()
pipe = tildapipe()
pipe.start()
pipe.pipeData.update(exampleCfg.dummyScanParameters)
pipe.pipeData.update(curVoltInd=0)
voltArray = np.zeros(exampleCfg.dummyScanParameters['nOfSteps'], dtype=np.uint32)
timeArray = np.zeros(exampleCfg.dummyScanParameters['nOfBins'], dtype=np.uint32)
scalerArray = np.zeros((exampleCfg.dummyScanParameters['nOfSteps'], exampleCfg.dummyScanParameters['nOfBins'], 8), dtype=np.uint32)
pipe.pipeData.update(voltArray=voltArray, timeArray=timeArray, scalerArray=scalerArray)
# print(len(trsExampleData))
# for i,j in enumerate(trsExampleData):
#     pipe.feed(j)


