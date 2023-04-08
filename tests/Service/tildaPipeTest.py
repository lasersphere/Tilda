"""

Created on '20.05.2015'

@author:'simkaufm'

"""

import os.path

import Tilda.Driver.DataAcquisitionFpga.TimeResolvedSequencerConfig as TRSConfig
import Tilda.Service.FileOperations.FolderAndFileHandling as filehand
import Tilda.Service.Scan.draftScanParameters as draftScan
from Tilda.Service.AnalysisAndDataHandling.tildaPipeline import TrsPipe


path = 'Tilda\\Extra\\exampleTRSRawData.py'
file = os.path.join(filehand.findTildaFolder(), path)
# trsExampleData = pickle.load(open(file, 'rb'))[1:]
# print(trsExampleData)
exampleCfg = TRSConfig
pipe = TrsPipe(draftScan.draftTrackPars)

pipe.start()


for i,j in enumerate(trsExampleData):
    pipe.feed(j)
    # pipe.save()
pipe.save()

#print(scalerArray)
# for i,j in enumerate(np.argwhere(pipe.pipeData['scalerArray'])):
#    print(pipe.pipeData['scalerArray'][j[0]][j[1]], j)
# print((pipe.pipeData['scalerArray'][0:][0:]))
#for i,j in enumerate(np.argwhere(pipe.pipeData['scalerArray'][:][:][0])):
#    print(pipe.pipeData['scalerArray'][j[0]][j[1]], j)

