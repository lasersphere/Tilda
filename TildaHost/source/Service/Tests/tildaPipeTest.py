"""

Created on '20.05.2015'

@author:'simkaufm'

"""

import pickle
from Service.AnalysisAndDataHandling.tildaPipeline import tildapipe



file = 'D:\\Workspace\\PyCharm\\Tilda\\TildaHost\\source\\Scratch\\exampleTRSRawData.py'
trsExampleData = pickle.load(open(file, 'rb'))[1:]
# print(trsExampleData)
pipe = tildapipe()
pipe.start()

print(len(trsExampleData))
for i,j in enumerate(trsExampleData):
    pipe.feed(j)


