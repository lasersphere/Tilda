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
