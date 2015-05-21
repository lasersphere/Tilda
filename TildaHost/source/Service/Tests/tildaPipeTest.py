__author__ = 'noertert'

import pickle

file = 'D:\\Workspace\\PyCharm\\Tilda\\TildaHost\\source\\Scratch\\exampleTRSRawData.py'
trsExampleData = pickle.load(open(file, 'rb'))[1:]
# print(trsExampleData)


