"""
Created on 

@author: simkaufm

Module Description: automatically created with the CApiAnalyser
"""

import ctypes


'''Bitfile Signature:'''
bitfileSignature = '1927BCDAF87BC3C5B46E16D2F712C497'
'''Bitfile Path:'''
bitfilePath = 'D:\\Workspace\\PyCharm\\Tilda\\TildaTarget\\bin\\SimpleCounter\\NiFpga_SimpleCounterV100.lvbitx'
'''FPGA Resource:'''
fpgaResource = 'Rio1'
'''Indicators:'''
postAccOffsetVoltState = {'ref': 0x8122, 'val': ctypes.c_ubyte(), 'ctr': False}
DacState = {'ref': 0x8112, 'val': ctypes.c_uint(), 'ctr': False}
actDACRegister = {'ref': 0x8114, 'val': ctypes.c_ulong(), 'ctr': False}
'''Controls:'''
postAccOffsetVoltControl = {'ref': 0x811E, 'val': ctypes.c_ubyte(), 'ctr': True}
DacStateCmdByHost = {'ref': 0x811A, 'val': ctypes.c_uint(), 'ctr': True}
setDACRegister = {'ref': 0x810C, 'val': ctypes.c_ulong(), 'ctr': True}
'''TargetToHostFifos:'''
transferToHost = {'ref': 0, 'val': ctypes.c_ulong(), 'ctr': False}


''' Hand filled '''
transferToHostReqEle = 50000

dacState = {'init': 0, 'idle': 1, 'setVolt': 2, 'error': 3}
postAccOffsetVoltStateDict = {'Kepco': 0, 'Heinzinger1': 1, 'Heinzinger2': 2, 'Heinzinger3': 3, 'loading': 4}
