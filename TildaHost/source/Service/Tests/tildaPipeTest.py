"""

Created on '20.05.2015'

@author:'simkaufm'

"""

import pickle
from Service.AnalysisAndDataHandling.tildaPipeline import tildapipe

from polliPipe.pipeline import Pipeline

file = 'D:\\Workspace\\PyCharm\\Tilda\\TildaHost\\source\\Scratch\\exampleTRSRawData.py'
trsExampleData = pickle.load(open(file, 'rb'))[1:]
# print(trsExampleData)
pipe = Pipeline(tildapipe())
pipe.start()
print(tildapipe().id)
print(len(trsExampleData))
for i,j in enumerate(trsExampleData):
    pipe.feed(j)


