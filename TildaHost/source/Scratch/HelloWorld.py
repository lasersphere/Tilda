'''
Created on 20.11.2014

@author: noertert
'''
# import ctypes
import numpy as np
# print('Hello World!')
#
# DACQuWriteTimeout = {'ref': 0x8116, 'val': ctypes.c_bool(), 'ctr': False}
#
# def func(dict):
#     print(dict['val'].value)
#     print(dict['ref'])
#
# func(DACQuWriteTimeout)

#
# print(len((ctypes.c_ulong * 0)()))
# print(int('00000010000000000000', 2))
# print(int('00000000000001000000', 2))

nOfBins = 100
nOfSteps = 20
testmatrix = np.zeros((nOfBins, nOfSteps, 8), dtype=np.int)
print(testmatrix)
print(type(testmatrix[0, 0, 0]))


#old main stuff:
#
# from Driver.DataAcquisitionFpga.TimeResolvedSequencer import TimeResolvedSequencer
# from Service.Formating import Formatter
# import time
#
# class Main():
#     def __init__(self):
#         self.trs = TimeResolvedSequencer()
#         self.form = Formatter()
#         self.fullData = []
#
#     def measureOneTrack(self, scanpars):
#         self.trs.measureTrack(scanpars)
#         while self.trs.getSeqState() == self.trs.TrsCfg.seqState['measureTrack']:
#             result = self.trs.getData()
#             if result['nOfEle'] == 0:
#                 break
#             else:
#                 print(result)
#                 newData = [self.form.integerSplitHeaderInfo(result['newData'][i]) for i in range(len(result['newData']))]
#                 print(newData)
#                 self.fullData.append(newData)
#                 time.sleep(0.4)
#         print(self.fullData)
#
#
#
# maininst = Main()
# print(maininst.measureOneTrack(maininst.trs.TrsCfg.dummyScanParameters))
#
# # print(format(814743552, '032b'))
# # print(len(bin(814743552)[2:]))